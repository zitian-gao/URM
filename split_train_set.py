"""Split a built train split into pretrain and RL subsets by puzzle.

The script expects the dataset layout produced by ARC augmentation (``train``
folder containing ``<set>__<field>.npy`` files plus a ``dataset.json``
metadata file). It shuffles puzzles, carves off a configurable ratio for the
pretrain subset, and writes two train directories for PPO pretraining and
RL fine-tuning.

Example:
    python scripts/split_train_set.py \
        --data-path /volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000 \
        --pretrain-path /volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000/ppo-pretrain \
        --rl-path /volume/pt-train/users/ztgao/loop_arcagi/data/arc-aug-1000/ppo-rl \
        --pretrain-ratio 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

DATA_FIELDS = (
    "inputs",
    "labels",
    "puzzle_identifiers",
    "puzzle_indices",
    "group_indices",
)


def _load_train_set(train_dir: Path, set_name: str) -> Dict[str, np.ndarray]:
    """Load all dataset arrays for a given set name from the train split."""
    arrays: Dict[str, np.ndarray] = {}
    for field in DATA_FIELDS:
        arrays[field] = np.load(train_dir / f"{set_name}__{field}.npy")
    return arrays


def _rebuild_subset(arrays: Dict[str, np.ndarray], puzzle_ids: Sequence[int]) -> Dict[str, np.ndarray]:
    """Construct a dataset subset containing only the selected puzzles."""
    puzzle_starts = arrays["puzzle_indices"]

    kept_inputs = []
    kept_labels = []
    kept_identifiers = []
    new_puzzle_indices = [0]

    for puzzle_id in puzzle_ids:
        start = int(puzzle_starts[puzzle_id])
        end = int(puzzle_starts[puzzle_id + 1])

        kept_inputs.append(arrays["inputs"][start:end])
        kept_labels.append(arrays["labels"][start:end])
        kept_identifiers.append(arrays["puzzle_identifiers"][puzzle_id])
        new_puzzle_indices.append(new_puzzle_indices[-1] + (end - start))

    if kept_inputs:
        inputs = np.concatenate(kept_inputs, axis=0)
        labels = np.concatenate(kept_labels, axis=0)
    else:
        inputs = arrays["inputs"][:0].copy()
        labels = arrays["labels"][:0].copy()

    puzzle_identifiers = np.asarray(kept_identifiers, dtype=arrays["puzzle_identifiers"].dtype)
    puzzle_indices = np.asarray(new_puzzle_indices, dtype=arrays["puzzle_indices"].dtype)
    group_indices = np.arange(puzzle_identifiers.size + 1, dtype=arrays["group_indices"].dtype)

    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }


def _save_subset(output_dir: Path, set_name: str, arrays: Dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field, values in arrays.items():
        np.save(output_dir / f"{set_name}__{field}.npy", values)


def split_train(data_path: Path, pretrain_path: Path, rl_path: Path, pretrain_ratio: float, seed: int) -> None:
    train_dir = data_path / "train"
    metadata_path = train_dir / "dataset.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata at {metadata_path}")

    if not 0.0 < pretrain_ratio < 1.0:
        raise ValueError("pretrain-ratio must be between 0 and 1 (exclusive)")

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    sets = metadata.get("sets", [])
    if not sets:
        raise ValueError("No set names listed in dataset metadata.")

    rng = np.random.default_rng(seed)

    for set_name in sets:
        arrays = _load_train_set(train_dir, set_name)

        num_puzzles = arrays["puzzle_indices"].size - 1
        order = np.arange(num_puzzles)
        rng.shuffle(order)

        cutoff = max(1, int(num_puzzles * pretrain_ratio))
        cutoff = min(cutoff, num_puzzles - 1)

        pretrain_ids = order[:cutoff]
        rl_ids = order[cutoff:]

        pretrain_split = _rebuild_subset(arrays, pretrain_ids)
        rl_split = _rebuild_subset(arrays, rl_ids)

        _save_subset(pretrain_path / "train", set_name, pretrain_split)
        _save_subset(rl_path / "train", set_name, rl_split)

    (pretrain_path / "train").mkdir(parents=True, exist_ok=True)
    (rl_path / "train").mkdir(parents=True, exist_ok=True)

    with (pretrain_path / "train" / "dataset.json").open("w") as f:
        json.dump(metadata, f)
    with (rl_path / "train" / "dataset.json").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a train split into PPO pretrain and RL subsets.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the source dataset (expects a 'train' directory)")
    parser.add_argument("--pretrain-path", type=Path, required=True, help="Where to write the pretrain subset")
    parser.add_argument("--rl-path", type=Path, required=True, help="Where to write the RL subset")
    parser.add_argument("--pretrain-ratio", type=float, default=0.8, help="Fraction of puzzles assigned to pretraining (default: 0.8)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for puzzle shuffling")
    args = parser.parse_args()

    split_train(
        data_path=args.data_path,
        pretrain_path=args.pretrain_path,
        rl_path=args.rl_path,
        pretrain_ratio=args.pretrain_ratio,
        seed=args.seed,
    )