#!/usr/bin/env python3
"""
Generate attention maps for two trained checkpoints (A/B) on the same test samples,
and visualize them side-by-side (A on the left, B on the right).
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch

from evaluate_trained_model import load_config_from_checkpoint
from models.layers import Attention, apply_rotary_pos_emb
from pretrain import PretrainConfig, create_dataloader
from utils import load_model_class

matplotlib.use("Agg")


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize attention maps for two checkpoints.")
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="attn_maps_ab")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(arg_device: Optional[str]) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_attention_tensors(
    maps_a,
    maps_b,
    puzzle_ids,
    output_dir: Path,
    side_length: int,
    checkpoint_a_path: Path,
    checkpoint_b_path: Path,
    step_a: int,
    step_b: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    attn_a = torch.stack([t.to(torch.float32).cpu() for t in maps_a], dim=0)
    attn_b = torch.stack([t.to(torch.float32).cpu() for t in maps_b], dim=0)
    attn_diff = attn_a - attn_b

    payload = {
        "attention_a": attn_a,              # [N, H, W]
        "attention_b": attn_b,
        "attention_diff": attn_diff,
        "puzzle_ids": puzzle_ids,
        "side_length": side_length,
        "checkpoint_a": {
            "path": str(checkpoint_a_path),
            "step": step_a,
        },
        "checkpoint_b": {
            "path": str(checkpoint_b_path),
            "step": step_b,
        },
    }

    torch.save(payload, output_dir / "attention_maps.pt")


# ============================================================
# Model / Checkpoint
# ============================================================

def build_model(config: PretrainConfig, metadata, device: torch.device) -> torch.nn.Module:
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore[attr-defined]
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    loss_kwargs = getattr(config.arch.loss, "__pydantic_extra__", {}) or {}
    model = loss_head_cls(model, **loss_kwargs)
    model.to(device)

    if device.type == "cuda" and "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model, dynamic=False)

    return model



def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = None
    step = checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0

    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

        if state_dict is None and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

    if state_dict is None:
        state_dict = checkpoint
    # Some checkpoints saved from torch.compile attach an "_orig_mod." prefix; strip it if present.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        stripped = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "", 1) if key.startswith("_orig_mod.") else key
            stripped[new_key] = value
        state_dict = stripped

    def _resize_puzzle_embedding_if_needed() -> None:
        inner = getattr(getattr(model, "model", None), "inner", None)
        if inner is None or not hasattr(inner, "puzzle_emb"):
            return

        expected = getattr(inner.puzzle_emb, "weights", None)
        if expected is None:
            return

        expected_shape = expected.shape
        candidate_keys = [
            "model.inner.puzzle_emb.weights",
            "_orig_mod.model.inner.puzzle_emb.weights",
        ]
        for key in candidate_keys:
            if key in state_dict:
                tensor = state_dict[key]
                if tensor.shape != expected_shape:
                    # Reinitialize by mean to preserve scale.
                    state_dict[key] = (
                        torch.mean(tensor, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
                break

    _resize_puzzle_embedding_if_needed()

    load_result = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = load_result
    if missing:
        print(f"Warning: missing keys during checkpoint load: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys during checkpoint load: {unexpected}")
    return {"step": step}


# ============================================================
# Attention Recorder
# ============================================================

class AttentionRecorder:
    def __init__(self, model: torch.nn.Module, puzzle_emb_len: int):
        self.puzzle_emb_len = puzzle_emb_len
        self.handles = []
        self._token_sum = None
        self._call_count = 0

        for module in model.modules():
            if isinstance(module, Attention):
                self.handles.append(
                    module.register_forward_hook(self._hook_fn, with_kwargs=True)
                )

    def reset(self):
        self._token_sum = None
        self._call_count = 0

    def close(self):
        for h in self.handles:
            h.remove()

    def _hook_fn(self, module: Attention, args: Tuple, *rest):
        kwargs = {}
        if rest and isinstance(rest[0], dict):
            kwargs = rest[0]

        cos_sin = kwargs.get("cos_sin")
        hidden_states = kwargs.get("hidden_states")

        # Fallback to positional args (defensive)
        if hidden_states is None:
            if len(args) >= 2:
                hidden_states = args[1]
            elif len(args) == 1:
                hidden_states = args[0]

        if hidden_states is None:
            return  # cannot proceed safely

        batch_size, seq_len, _ = hidden_states.shape

        qkv = module.qkv_proj(hidden_states)
        qkv = qkv.view(
            batch_size,
            seq_len,
            module.num_heads + 2 * module.num_key_value_heads,
            module.head_dim,
        )

        query = qkv[:, :, : module.num_heads]
        key = qkv[:, :, module.num_heads : module.num_heads + module.num_key_value_heads]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if module.num_key_value_heads != module.num_heads:
            repeat = module.num_heads // module.num_key_value_heads
            key = key.repeat_interleave(repeat, dim=2)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(module.head_dim)
        attn = torch.softmax(scores, dim=-1)

        attn = attn[:, :, self.puzzle_emb_len :, self.puzzle_emb_len :]
        token_focus = attn.mean(dim=1).mean(dim=1).detach().cpu()

        if self._token_sum is None:
            self._token_sum = token_focus
        else:
            self._token_sum += token_focus

        self._call_count += 1

    def get_average(self) -> torch.Tensor:
        return self._token_sum / self._call_count


def collect_test_samples(loader, metadata, num_samples):
    out = {"inputs": [], "labels": [], "puzzle_identifiers": []}
    remain = num_samples

    for _, batch, _ in loader:
        mask = batch["puzzle_identifiers"] != metadata.blank_identifier_id
        idx = mask.nonzero(as_tuple=True)[0][:remain]
        for k in out:
            out[k].append(batch[k][idx])
        remain -= len(idx)
        if remain <= 0:
            break

    return {k: torch.cat(v)[:num_samples] for k, v in out.items()}

def save_two_puzzles_pdf(
    maps_a,
    maps_b,
    puzzle_ids,
    puzzle_id_left: int,
    puzzle_id_right: int,
    output_path: Path,
    title_a: str = "No Conv",
    title_b: str = "With Conv",
):
    # 找 index
    try:
        idx0 = puzzle_ids.index(puzzle_id_left)
        idx1 = puzzle_ids.index(puzzle_id_right)
    except ValueError as e:
        raise ValueError("Specified puzzle_id not found in puzzle_ids") from e

    a0, b0 = maps_a[idx0], maps_b[idx0]
    a1, b1 = maps_a[idx1], maps_b[idx1]

    vmin = min(a0.min(), b0.min(), a1.min(), b1.min())
    vmax = max(a0.max(), b0.max(), a1.max(), b1.max())

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.15,
    )

    # --- Row 0: puzzle_id_left ---
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[:, 2])  # colorbar 跨两行

    im = ax00.imshow(a0.to(torch.float32).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="viridis")
    ax01.imshow(b0.to(torch.float32).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="viridis")

    ax00.set_title(title_a)
    ax01.set_title(title_b)
    ax00.set_ylabel(f"Puzzle {puzzle_id_left}", fontsize=11)

    # --- Row 1: puzzle_id_right ---
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    ax10.imshow(a1.to(torch.float32).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="viridis")
    ax11.imshow(b1.to(torch.float32).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="viridis")

    ax10.set_ylabel(f"Puzzle {puzzle_id_right}", fontsize=11)

    # clean axes
    for ax in [ax00, ax01, ax10, ax11]:
        ax.axis("off")

    fig.colorbar(im, cax=cax)

    fig.suptitle(
        f"Attention Comparison for Two Puzzles",
        fontsize=14,
        y=0.97,
    )

    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def chunk(batch, s, e):
    return {k: v[s:e] for k, v in batch.items()}

def generate_attention_maps(
    checkpoint_path,
    config,
    metadata,
    samples,
    device,
    seq_side,
    inference_batch_size,
):
    model = build_model(config, metadata, device)
    step = load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    puzzle_emb_len = getattr(
        getattr(getattr(model, "model", None), "inner", None),
        "puzzle_emb_len",
        0,
    ) or 0

    recorder = AttentionRecorder(model, puzzle_emb_len)
    maps, ids = [], []

    with torch.inference_mode():
        for s in range(0, samples["inputs"].size(0), inference_batch_size):
            b = chunk(samples, s, s + inference_batch_size)
            b = {k: v.to(device) for k, v in b.items()}

            recorder.reset()
            with torch.device(device):
                carry = model.initial_carry(b)

            while True:
                carry, *_ , done = model(return_keys=set(), carry=carry, batch=b)
                if done:
                    break

            avg = recorder.get_average()
            for i in range(avg.size(0)):
                maps.append(avg[i].reshape(seq_side, seq_side))
                ids.append(int(b["puzzle_identifiers"][i]))

    recorder.close()
    return maps, ids, step


def save_side_by_side(
    maps_a, maps_b, puzzle_ids, output_dir, step_a, step_b
):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for i, (a, b, pid) in enumerate(zip(maps_a, maps_b, puzzle_ids)):
        fig = plt.figure(figsize=(9, 4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.2)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        vmin = min(a.min(), b.min())
        vmax = max(a.max(), b.max())

        im0 = ax0.imshow(a.to(torch.float32).cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        im1 = ax1.imshow(b.to(torch.float32).cpu().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)

        ax0.set_title(f"No Conv")
        ax1.set_title(f"With Conv")
        fig.suptitle(
            f"Puzzle ID {pid}",
            fontsize=12,
            y=0.98,
        )

        ax0.axis("off")
        ax1.axis("off")

        fig.colorbar(im0, cax=cax)

        fig.subplots_adjust(top=0.85)
        path = output_dir / f"sample_{i:03d}.pdf"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        summary.append({"index": i, "puzzle_id": pid, "path": path.name})

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    config = load_config_from_checkpoint(Path(args.checkpoint_a))
    config.data_path = "/volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000"
    config.global_batch_size = args.batch_size

    loader, metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    samples = collect_test_samples(loader, metadata, args.num_samples)
    seq_len = samples["inputs"].shape[1]
    side = int(math.sqrt(seq_len))

    maps_a, ids_a, step_a = generate_attention_maps(
        Path(args.checkpoint_a),
        config,
        metadata,
        samples,
        device,
        side,
        args.inference_batch_size,
    )

    maps_b, ids_b, step_b = generate_attention_maps(
        Path(args.checkpoint_b),
        config,
        metadata,
        samples,
        device,
        side,
        args.inference_batch_size,
    )

    assert ids_a == ids_b

    save_side_by_side(
        maps_a,
        maps_b,
        ids_a,
        Path(args.output_dir),
        step_a,
        step_b,
    )
    
    save_attention_tensors(
        maps_a=maps_a,
        maps_b=maps_b,
        puzzle_ids=ids_a,
        output_dir=Path(args.output_dir),
        side_length=side,
        checkpoint_a_path=Path(args.checkpoint_a),
        checkpoint_b_path=Path(args.checkpoint_b),
        step_a=step_a,
        step_b=step_b,
    )

    save_two_puzzles_pdf(
        maps_a=maps_a,
        maps_b=maps_b,
        puzzle_ids=ids_a,
        puzzle_id_left=508256,
        puzzle_id_right=508258,
        output_path=Path(args.output_dir) / "puzzle_508256_508258.pdf",
    )

    print(f"Saved {len(maps_a)} side-by-side attention maps to {args.output_dir}")


if __name__ == "__main__":
    main()
