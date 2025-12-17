#!/usr/bin/env python3
"""
Augmentation Ablation ARC Evaluator

Inherits from ARC and overrides only the result() method to sample
max_augmentations_per_task from each base task's collected predictions.

This preserves 100% of the existing evaluation logic while only changing
the sampling step, making it maximally trustworthy.
"""

import random
from typing import Dict, Optional, Sequence

import torch

from evaluators.arc import ARC
from data.build_arc_dataset import grid_hash, arc_grid_to_np


class AugmentationAblationARC(ARC):
    """
    ARC evaluator that samples limited augmentations per base task.

    Uses identical logic to parent class but samples predictions in result().
    At max_augmentations_per_task=1000, behaves exactly like parent class.
    """

    def __init__(
        self,
        data_path: str,
        eval_metadata,
        submission_K: int = 2,
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
        aggregated_voting: bool = True,
        max_augmentations_per_task: int = 1000,
        random_seed: int = 42,
    ):
        super().__init__(data_path, eval_metadata, submission_K, pass_Ks, aggregated_voting)

        self.max_augmentations_per_task = max_augmentations_per_task
        self.random_seed = random_seed

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Clean augmentation sampling at the voting level.

        Sample predictions per test example before voting, which is the right granularity
        for answering "how many augmentations do we need for good performance?"
        """
        # Gather predictions to rank 0 for voting (same as parent)
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        torch.distributed.gather_object(
            (self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group
        )

        # Rank 0 logic
        if rank != 0:
            return

        # Set up sampling
        rng = random.Random(self.random_seed)
        examples_sampled = 0
        total_predictions_before = 0
        total_predictions_after = 0

        print(f"Sampling predictions: max {self.max_augmentations_per_task} per test example")

        # Identical to parent class logic, with sampling injection
        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        # Detailed logging for first few examples
        detailed_log_count = 0

        for name, puzzle in self.test_puzzles.items():
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]

            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                # === SAMPLING INJECTION: Collect then sample predictions ===
                all_predictions = []
                for hmap, preds in global_hmap_preds:  # type: ignore
                    # Fix bug in original: should be [] not {}
                    for h, q in preds.get(name, {}).get(input_hash, []):
                        all_predictions.append((h, q))

                total_predictions_before += len(all_predictions)

                # Sample if needed
                if len(all_predictions) > self.max_augmentations_per_task:
                    sampled_predictions = rng.sample(all_predictions, self.max_augmentations_per_task)
                    examples_sampled += 1
                else:
                    sampled_predictions = all_predictions

                total_predictions_after += len(sampled_predictions)

                # Detailed logging for first 3 examples
                if detailed_log_count < 3:
                    print(
                        f"  Example {detailed_log_count + 1} ({name}): "
                        f"{len(all_predictions)}→{len(sampled_predictions)} predictions"
                    )
                    if len(all_predictions) != len(sampled_predictions):
                        orig_q_values = [q for _, q in all_predictions]
                        sampled_q_values = [q for _, q in sampled_predictions]
                        print(
                            f"    Original Q-values: min={min(orig_q_values):.3f}, max={max(orig_q_values):.3f}"
                        )
                        print(
                            f"    Sampled Q-values: min={min(sampled_q_values):.3f}, max={max(sampled_q_values):.3f}"
                        )
                    detailed_log_count += 1
                # === END SAMPLING INJECTION ===

                # Original voting logic (exactly as parent class)
                p_map = {}
                for h, q in sampled_predictions:  # Use sampled instead of original predictions
                    p_map.setdefault(h, [0, 0])
                    p_map[h][0] += 1
                    p_map[h][1] += q

                if not len(p_map):
                    print(f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                # Note: Pass@K works correctly even when k > actual predictions available
                # p_map[:k] automatically uses min(k, len(p_map)) predictions
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash

                    num_test_correct[i] += ok

                # Query grids
                pred_grids = []
                for h, stats in p_map[: self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append(
                    {f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)}
                )

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        # Save submission (same as parent)
        if save_path is not None:
            import json
            import os

            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result with additional metadata
        metrics = {
            f"{self.__class__.__name__}/pass@{k}": correct[i] / len(self.test_puzzles)
            for i, k in enumerate(self.pass_Ks)
        }

        # Add ablation metadata
        metrics[f"{self.__class__.__name__}/max_augmentations_per_task"] = float(
            self.max_augmentations_per_task
        )

        # Print summary statistics
        print(f"\n=== Augmentation Sampling Summary ===")
        print(f"Max augmentations per example: {self.max_augmentations_per_task}")
        print(f"Examples with sampling applied: {examples_sampled}")
        print(f"Total predictions: {total_predictions_before} → {total_predictions_after}")

        reduction_pct = 100 * (1 - total_predictions_after / max(total_predictions_before, 1))
        print(f"Prediction reduction: {reduction_pct:.1f}%")

        print(f"\nPass@K Results:")
        for i, k in enumerate(self.pass_Ks):
            print(f"  Pass@{k}: {correct[i] / len(self.test_puzzles):.4f}")

        print(f"=== End Summary ===\n")

        return metrics
