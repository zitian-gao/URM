#!/usr/bin/env python3
"""
Standalone evaluation script for trained HRM models.
Loads a checkpoint and evaluates it on a specified dataset.
Supports both single and multi-GPU evaluation.

Usage:
# Single GPU
python evaluate_trained_model.py \
    --checkpoint-path "/volume/pt-train/users/ztgao/loop_arcagi/checkpoints/URM-SwiGLU/step_518050.pt" \
    --data-path /volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000 \
    --output-dir eval_results
    
# Multi-GPU
torchrun --nproc-per-node 8 evaluate_trained_model.py \
    --checkpoint-path "/volume/pt-train/users/ztgao/loop_arcagi/checkpoints/Arc-aug-1000 ACT-torch/LoopedTransformerV6 zippy-bull/step_23825.pt" \
    --data-path /volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000 \
    --output-dir eval_results
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.distributed as dist
import numpy as np
import wandb

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from pretrain import (
    PretrainConfig,
    create_dataloader, 
    create_model,
    create_evaluators,
    evaluate,
    load_checkpoint,
    TrainState
)
from utils import load_model_class


def setup_distributed():
    """Initialize distributed training if in distributed environment."""
    rank = 0
    world_size = 1
    cpu_group = None
    
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed
        dist.init_process_group(backend="nccl")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group for evaluation
        cpu_group = dist.new_group(backend="gloo")
        assert dist.get_rank(cpu_group) == rank and dist.get_world_size(cpu_group) == world_size
    
    return rank, world_size, cpu_group


def load_config_from_checkpoint(checkpoint_path: Path) -> PretrainConfig:
    import yaml
    import re

    checkpoint_dir = checkpoint_path.parent
    cfg_path = checkpoint_dir / "all_config.yaml"
    raw = cfg_path.read_text()

    # -------- 1. clean YAML 从 beta1 开始（此部分可正常解析）--------
    idx = raw.find("\nbeta1:")
    if idx == -1:
        raise RuntimeError("Cannot locate clean YAML block (beta1:)")

    clean_yaml = raw[idx+1:]
    flat = yaml.safe_load(clean_yaml)

    # -------- 2. 从损坏 Hydra dump 中提取 arch 字段（你之前已验证可行）--------
    # 提取 loops
    m = re.search(r"\bloops:\s*(\d+)", raw)
    loops = int(m.group(1)) if m else 16

    # num_heads
    m = re.search(r"\bnum_heads:\s*(\d+)", raw)
    num_heads = int(m.group(1)) if m else 8

    # num_layers
    m = re.search(r"\bnum_layers:\s*(\d+)", raw)
    num_layers = int(m.group(1)) if m else 4

    # pos_encodings
    m = re.search(r"\bpos_encodings:\s*(\w+)", raw)
    pos_encodings = m.group(1) if m else "rope"

    # puzzle_emb_ndim
    m = re.search(r"\bpuzzle_emb_ndim:\s*(\d+)", raw)
    puzzle_emb_ndim = int(m.group(1)) if m else 512

    # hidden_size
    m = re.search(r"\bhidden_size:\s*(\d+)", raw)
    hidden_size = int(m.group(1)) if m else puzzle_emb_ndim

    # expansion
    m = re.search(r"\bexpansion:\s*(\d+)", raw)
    expansion = int(m.group(1)) if m else 4

    # H_cycles
    m = re.search(r"\bH_cycles:\s*(\d+)", raw)
    H_cycles = int(m.group(1)) if m else 2

    # L_cycles
    m = re.search(r"\bL_cycles:\s*(\d+)", raw)
    L_cycles = int(m.group(1)) if m else 6

    arch_name = "loop.v22@LoopedTransformer"

    # -------- arch.loss 字段无法从 clean 找到，填入正确值 --------
    arch_loss = {
        "loss_type": "stablemax_cross_entropy",
        "name": "losses@ACTLossHead"
    }

    # -------- 3. 构造 arch dict --------
    arch = {
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "expansion": expansion,
        "hidden_size": hidden_size,
        "loop_deltas": [0, 8],
        "loops": loops,
        "loss": arch_loss,
        "name": arch_name,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "pos_encodings": pos_encodings,
        "puzzle_emb_ndim": puzzle_emb_ndim,
    }

    # -------- 4. 合并到 flat 并构造 PretrainConfig --------
    flat["arch"] = arch

    return PretrainConfig(**flat)


def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    save_predictions: bool = False,
    loop_offsets: Optional[List[int]] = None,
):
    """
    Evaluate a trained model checkpoint on a specified dataset.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_path: Path to the dataset for evaluation
        output_dir: Directory to save evaluation results
        config_overrides: Optional config overrides
        wandb_project: Optional W&B project name
        wandb_run_name: Optional W&B run name
        save_predictions: Whether to save model predictions
    """
    # Setup distributed if needed
    rank, world_size, cpu_group = setup_distributed()
    
    # Load config from checkpoint
    checkpoint_path = Path(checkpoint_path)
    if rank == 0:
        print(f"Loading config from checkpoint: {checkpoint_path}")
    
    config = load_config_from_checkpoint(checkpoint_path)

    # Apply overrides
    config.checkpoint_path = str(checkpoint_path.parent)
    config.data_path = data_path
    
    if config_overrides:
        from pretrain import EvaluatorConfig  # Local import to avoid circular during type checking

        for key, value in config_overrides.items():
            if key == "arch" and isinstance(value, dict):
                # Handle nested arch config updates (e.g., halt_max_steps)
                for arch_key, arch_value in value.items():
                    if hasattr(config.arch, '__pydantic_extra__'):
                        config.arch.__pydantic_extra__[arch_key] = arch_value
                    else:
                        setattr(config.arch, arch_key, arch_value)
            elif key == "evaluators" and isinstance(value, list):
                # Convert evaluator override dicts back into EvaluatorConfig objects
                config.evaluators = [
                    cfg if isinstance(cfg, EvaluatorConfig) else EvaluatorConfig(**cfg) for cfg in value
                ]
            else:
                setattr(config, key, value)
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare loop extrapolation settings
    arch_params = getattr(config.arch, "__pydantic_extra__", {})
    base_loops = arch_params.get("loops") if isinstance(arch_params, dict) else None
    use_loop_extrapolation = loop_offsets is not None and len(loop_offsets) > 0

    if use_loop_extrapolation and base_loops is None:
        raise ValueError("Loop extrapolation requested, but the loaded config has no 'loops' parameter.")

    loop_runs: List[Tuple[Optional[int], PretrainConfig]] = []
    if use_loop_extrapolation:
        for offset in loop_offsets or []:
            loop_config = config.model_copy(deep=True)
            loop_config.arch.__pydantic_extra__["loops"] = base_loops + offset  # type: ignore[index]
            loop_runs.append((base_loops + offset, loop_config))
    else:
        loop_runs.append((base_loops, config))

    # Load dataset once (independent of loop setting)
    if rank == 0:
        print(f"Loading evaluation dataset from: {data_path}")

    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size
        )
    except FileNotFoundError as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
            print("Make sure the dataset exists and has a 'test' split")
        return
    
    # Load model weights once (rank 0) to reuse across loop runs
    if rank == 0:
        print(f"Loading checkpoint weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

            if state_dict is None and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

        if state_dict is None:
            state_dict = checkpoint

        step = checkpoint.get('step', 0)
    else:
        state_dict = None
        step = 0

    if world_size > 1:
        step_tensor = torch.tensor([step], device='cuda')
        dist.broadcast(step_tensor, src=0)
        step = step_tensor.item()

    loop_metrics: Dict[int, Dict[str, Any]] = {}

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    for loop_value, loop_config in loop_runs:
        if rank == 0:
            loop_desc = f"loops={loop_value}" if loop_value is not None else "loops=default"
            print(f"\n=== Evaluating with {loop_desc} ===")
            print("Creating model...")

        # Load model - we need to get training metadata for model creation
        try:
            train_loader, train_metadata = create_dataloader(
                loop_config,
                "train",
                test_set_mode=False,
                epochs_per_iter=1,
                global_batch_size=loop_config.global_batch_size,
                rank=rank,
                world_size=world_size
            )
        except FileNotFoundError:
            if rank == 0:
                print("No train split found, using eval metadata for model creation")
            train_metadata = eval_metadata

        model, _, _ = create_model(loop_config, train_metadata, rank=rank, world_size=world_size)

        if rank == 0:
            model.load_state_dict(state_dict, strict=True)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

        if rank == 0:
            print("Creating evaluators...")
        evaluators = create_evaluators(loop_config, eval_metadata)

        train_state = TrainState(
            model=model,
            optimizers=[],  # Not needed for evaluation
            optimizer_lrs=[],  # Not needed for evaluation
            carry=None,  # Will be initialized during evaluation
            step=step,
            total_steps=step + 1  # Just needs to be > step
        )

        model.eval()

        if rank == 0:
            print("Running evaluation...")
            print(f"Dataset has {len(eval_metadata.sets)} test sets")

        loop_output_dir = output_dir if not use_loop_extrapolation else output_dir / f"loops_{loop_value}"
        loop_output_dir.mkdir(parents=True, exist_ok=True)

        if save_predictions:
            loop_config.eval_save_outputs = ["inputs", "preds", "puzzle_identifiers"]

        if rank == 0 and wandb_project:
            wandb_run = wandb_run_name or f"eval_{checkpoint_path.stem}"
            if loop_value is not None:
                wandb_run = f"{wandb_run}_loops{loop_value}"

            wandb.init(
                project=wandb_project,
                name=wandb_run,
                config=OmegaConf.to_container(OmegaConf.create(loop_config.__dict__)),
                dir=str(loop_output_dir)
            )

        metrics = evaluate(
            loop_config,
            train_state,
            eval_loader,
            eval_metadata,
            evaluators,
            rank=rank,
            world_size=world_size,
            cpu_group=cpu_group
        )

        if rank == 0 and metrics is not None:
            serializable_metrics = convert_to_serializable(metrics)

            metrics_file = loop_output_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)

            print("\nEvaluation Results:")
            print("=" * 50)
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue:.4f}")
                else:
                    print(f"{key}: {value:.4f}")

            print(f"\nResults saved to: {loop_output_dir}")

            if use_loop_extrapolation and loop_value is not None:
                loop_metrics[loop_value] = serializable_metrics

            if wandb.run:
                wandb.log(metrics)
                wandb.finish()

        if world_size > 1:
            dist.barrier()

    if use_loop_extrapolation and rank == 0 and len(loop_metrics):
        summary_file = output_dir / "loop_extrapolation_metrics.json"
        with open(summary_file, 'w') as f:
            json.dump(loop_metrics, f, indent=2)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HRM model checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Global batch size for evaluation"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (optional)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions"
    )
    parser.add_argument(
        "--submission-k",
        type=int,
        default=2,
        help="Number of predictions per puzzle for submission"
    )
    parser.add_argument(
        "--aggregated-voting",
        action="store_true",
        default=True,
        help="Use aggregated voting across augmentations"
    )
    parser.add_argument(
        "--loop-extrapolation",
        action="store_true",
        help="Evaluate the checkpoint with additional loop counts beyond training",
    )
    parser.add_argument(
        "--loop-offsets",
        type=str,
        default="0,1,3,5,7,9",
        help="Comma-separated list of loop count offsets to test during extrapolation",
    )

    args = parser.parse_args()

    # pass_ks = [1, 2, 5, 10, 100, 1000]
    pass_ks = [1]
    loop_offsets = None
    # if args.loop_extrapolation:
    #     loop_offsets = [int(item) for item in args.loop_offsets.split(',') if item.strip()]
    #     pass_ks = [1]
        # pass_ks = [1, 2, 5, 10, 100, 1000]

    config_overrides = {
        "global_batch_size": args.batch_size,
        "evaluators": [
            {
                "name": "ARC",
                "submission_K": args.submission_k,
                "aggregated_voting": args.aggregated_voting,
                "pass_Ks": pass_ks
            }
        ]
    }

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_overrides=config_overrides,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_predictions=args.save_predictions,
        loop_offsets=loop_offsets
    )


if __name__ == "__main__":
    main()