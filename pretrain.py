from typing import Optional, Any, Sequence, List, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import re
import copy
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2
# from adam_atan2_pytorch import MuonAdamAtan2
from models.muon import Muon
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    target_q_update_every: int

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Gradient accumulation
    grad_accum_steps: int = 1

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    load_checkpoint: Optional[str] = None
    load_strict: bool = True
    load_optimizer_state: bool = True

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

    loop_deltas: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999

    use_muon: bool = False



@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    accum_step: int = 0


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed, dataset_path=config.data_path, rank=rank, num_replicas=world_size, **kwargs
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=1, prefetch_factor=8, pin_memory=True, persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        model_config = getattr(getattr(model, "model", None), "config", None)
        should_compile = (
            "DISABLE_COMPILE" not in os.environ
            and (model_config is None or not getattr(model_config, "profile", False))
        )
        if should_compile:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    if config.use_muon:
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        muon_params = [p for p in model.parameters() if p.ndim == 2]

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            Muon([
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": 1e-4,
                },
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": 1e-4,
                    "weight_decay": 0.1,
                    "adamw_betas": (0.9, 0.95),
                    "adamw_eps": 1e-8,
                },
            ]),
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]

    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    )


def init_train_state(
    config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int
):
    # Estimated total training steps
    effective_gbs = config.global_batch_size * max(1, config.grad_accum_steps)
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / effective_gbs
    )

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    train_state = TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )

    load_checkpoint(train_state, config, rank)

    return train_state


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "step": train_state.step,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_states": [optim.state_dict() for optim in train_state.optimizers],
    }

    state["rng_state"] = torch.random.get_rng_state()
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(state, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))


def _resolve_checkpoint_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        pattern = re.compile(r"step_(\\d+)(?:\\.pt)?$")
        candidates: List[Tuple[int, str]] = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(path, file_name)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    return None


def load_config_from_checkpoint_path(path: str) -> Optional[PretrainConfig]:
    """Load a saved config from a checkpoint directory, if present."""

    resolved_path = _resolve_checkpoint_path(path)
    checkpoint_dir = Path(resolved_path if resolved_path is not None else path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    def _load_candidate(candidate: Path) -> Optional[PretrainConfig]:
        if not candidate.exists():
            return None

        # Prefer OmegaConf so we can parse Hydra-specific tags written during training.
        try:
            conf = OmegaConf.load(candidate)
            # Convert to a plain container so pydantic can consume it.
            as_dict = OmegaConf.to_container(conf, resolve=True)
            if isinstance(as_dict, dict):
                return PretrainConfig(**as_dict)
        except Exception:
            pass

        # Fallback to a plain YAML load if OmegaConf parsing fails for any reason.
        try:
            with open(candidate, "r") as f:
                config_dict = yaml.safe_load(f)
            if isinstance(config_dict, dict):
                return PretrainConfig(**config_dict)
        except Exception:
            pass

        return None

    for candidate in [checkpoint_dir / "all_config.yaml", checkpoint_dir / ".hydra" / "config.yaml"]:
        loaded = _load_candidate(candidate)
        if loaded is not None:
            return loaded

    return None


def _resize_puzzle_embedding_if_needed(model: nn.Module, state_dict: dict):
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
    if puzzle_emb_name in state_dict:
        puzzle_emb = state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(
                f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}"
            )

            # Re-initialize using mean
            state_dict[puzzle_emb_name] = (
                torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
            )


def load_checkpoint(train_state: TrainState, config: PretrainConfig, rank: int):
    load_path = config.load_checkpoint
    if load_path is None:
        return

    if load_path == "latest":
        if config.checkpoint_path is None:
            raise ValueError("Cannot load latest checkpoint without a checkpoint_path configured.")
        load_path = config.checkpoint_path

    resolved_path = _resolve_checkpoint_path(load_path)
    if resolved_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from '{load_path}'")

    if rank == 0:
        print(f"Loading checkpoint {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location="cuda")

    def _prepare_rng_state(state: Any, device: str | None) -> Any:
        """Ensure RNG state tensors are on the correct device and uint8 dtype."""

        if state is None:
            return None

        if isinstance(state, (list, tuple)):
            return [_prepare_rng_state(s, device) for s in state]

        tensor_state = torch.as_tensor(state, device=device)
        if tensor_state.dtype != torch.uint8:
            tensor_state = tensor_state.to(torch.uint8)

        return tensor_state

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        optimizer_states = checkpoint.get("optimizer_states")
        step = checkpoint.get("step")
        rng_state = checkpoint.get("rng_state")
        cuda_rng_state = checkpoint.get("cuda_rng_state")
    else:
        # Backwards compatibility with checkpoints that only contain model weights
        state_dict = checkpoint
        optimizer_states = None
        step = None
        rng_state = None
        cuda_rng_state = None

    _resize_puzzle_embedding_if_needed(train_state.model, state_dict)
    try:
        load_result = train_state.model.load_state_dict(
            state_dict, strict=config.load_strict, assign=True
        )
    except RuntimeError:
        # Re-raise with clearer guidance if strict loading was requested.
        raise

    if not config.load_strict and rank == 0:
        missing, unexpected = load_result
        if missing:
            print(f"Warning: missing keys during checkpoint load: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys during checkpoint load: {unexpected}")

    if optimizer_states is not None:
        if not config.load_optimizer_state:
            if rank == 0:
                print("Skipping optimizer state load because load_optimizer_state=False")
        elif len(optimizer_states) != len(train_state.optimizers):
            raise ValueError(
                "Checkpoint optimizer count does not match current configuration: "
                f"{len(optimizer_states)} vs {len(train_state.optimizers)}"
            )
        else:
            for optimizer, optimizer_state in zip(train_state.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)

    if step is not None:
        train_state.step = int(step)

    # Reset carry since we do not serialize it
    train_state.carry = None

    if rng_state is not None:
        normalized_rng_state = _prepare_rng_state(rng_state, device="cpu")
        # Older checkpoints should always store a single tensor here.
        if isinstance(normalized_rng_state, list):
            normalized_rng_state = normalized_rng_state[0]
        torch.random.set_rng_state(normalized_rng_state)

    if cuda_rng_state is not None and torch.cuda.is_available():
        normalized_cuda_state = _prepare_rng_state(cuda_rng_state, device="cpu")
        try:
            if isinstance(normalized_cuda_state, list):
                if len(normalized_cuda_state) != torch.cuda.device_count():
                    primary_state = normalized_cuda_state[0]
                    normalized_cuda_state = [
                        primary_state for _ in range(torch.cuda.device_count())
                    ]
                torch.cuda.set_rng_state_all(normalized_cuda_state)
            else:
                torch.cuda.set_rng_state(normalized_cuda_state)
        except RuntimeError:
            fallback_state = (
                normalized_cuda_state[0]
                if isinstance(normalized_cuda_state, list)
                else normalized_cuda_state
            )
            torch.cuda.set_rng_state(fallback_state)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        cls = load_model_class(cfg.name, "evaluators.")(
            data_path=config.data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
        )  # type: ignore
        evaluators.append(cls)

    return evaluators


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
    if train_state.step >= train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    compute_target_q = train_state.step % config.target_q_update_every == 0
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[], compute_target_q=compute_target_q
    )

    loss_scale = 1.0 / (global_batch_size * accum_steps)
    (loss_scale * loss).backward()
    train_state.accum_step += 1

    should_step = train_state.accum_step % accum_steps == 0
    if not should_step:
        return

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                grad = torch.zeros_like(param)
            dist.all_reduce(grad)

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    train_state.step += 1
    train_state.accum_step = 0

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics.get("count", 0), 1)  # Avoid NaNs

            def _normalize_metric(key: str, value: float) -> float:
                if key.startswith("profile/"):
                    return value / world_size
                if key.endswith("loss"):
                    return value / global_batch_size
                return value / count

            reduced_metrics = {f"train/{k}": _normalize_metric(k, v) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def save_code_and_config(config: PretrainConfig, save_dir: str):
    import os, json
    import yaml

    cfg_path = os.path.join(save_dir, "config.yaml")
    json_path = os.path.join(save_dir, "config.json")

    config_dict = json.loads(config.model_dump_json())

    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)

    except Exception as e:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "# Failed to write config as YAML, wrote config.json instead.\n"
                f"# Error: {type(e).__name__}: {e}\n"
            )


def _get_loop_config(model: nn.Module):
    inner_model = getattr(model, "model", None)
    model_config = getattr(inner_model, "config", None)
    if model_config is None or not hasattr(model_config, "loops"):
        return None

    return model_config


def _prefix_metrics(metrics: Any, prefix: str):
    if metrics is None:
        return {}

    prefixed = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                prefixed[f"{prefix}/{key}/{sub_key}"] = sub_value
        else:
            prefixed[f"{prefix}/{key}"] = value

    return prefixed


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        config.project_name = "arcagi"

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    # Train loader
    train_epochs_per_iter = config.eval_interval
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    
    # Eval loader
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
        # Evaluators
        evaluators = create_evaluators(config, eval_metadata)
    except FileNotFoundError:
        print(f"eval metadata FileNotFoundError")
        eval_loader = eval_metadata = None
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    ema_helper = None
    if config.ema:
        if RANK == 0:
            print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if train_state.step > 0:
            progress_bar.update(train_state.step)

        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        if RANK == 0:
            count = 0
            for set_name, batch, global_batch_size in train_loader:
                count += 1
            print(f"_iter_id: {_iter_id}")
            print(f"train_epochs_per_iter: {train_epochs_per_iter}")
            print(f"total_iters: {total_iters}")
            print(f"train_loader len: {count}")
            print(f"Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()

        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE
            )

            # EMA update
            if config.ema and ema_helper is not None:
                ema_helper.update(train_state.model)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        ############ Evaluation
        if eval_loader is not None and eval_metadata is not None:
            # 选择用于评估的 train_state（EMA 或原始）
            if config.ema and ema_helper is not None:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            loop_config = _get_loop_config(train_state_eval.model)
            if loop_config is not None:
                original_loops = loop_config.loops
                if len(config.loop_deltas) == 0:
                    config.loop_deltas = [0, 8]
                else:
                    config.loop_deltas = [0]
            for delta in config.loop_deltas:
                if loop_config is not None:
                    loop_config.loops = original_loops + delta

                metrics = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                )
                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)

            if loop_config is not None:
                loop_config.loops = original_loops

            # 用完临时的 eval state 后可以丢掉，节省显存/内存
            if config.ema and ema_helper is not None and train_state_eval is not train_state:
                del train_state_eval

        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            if config.ema and ema_helper is not None:
                # 临时拷贝一个带 EMA 权重的 state 来保存
                ts_to_save = copy.deepcopy(train_state)
                ts_to_save.model = ema_helper.ema_copy(ts_to_save.model)
                save_train_state(config, ts_to_save)
                del ts_to_save
            else:
                save_train_state(config, train_state)


    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
