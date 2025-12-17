from typing import Dict, Sequence, Optional
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
import torch.distributed as dist

from data.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from data.common import PuzzleDatasetMetadata


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, submission_K: int = 2, pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Majority vote evaluation settings
        self.maj_sample_sizes = (10, 100, 1000, 10000)

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        q_values = None
        q_log_probs = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                        q_log_probs = F.logsigmoid(v.to(torch.float64)).cpu()
                    else:
                        outputs[k] = v.cpu()

        assert q_values is not None and q_log_probs is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input, pred, q, q_log_prob in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), q_values.numpy(), q_log_probs.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)

            input_hash = grid_hash(_inverse_fn(_crop(input)))
            
            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q), float(q_log_prob)))
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # Gather predictions to rank 0 for voting
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)
        
        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        maj_correct = {size: 0.0 for size in self.maj_sample_sizes}

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            maj_test_correct = {size: 0 for size in self.maj_sample_sizes}
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                p_map = {}
                pred_samples = []
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q, q_log_prob in preds.get(name, {}).get(input_hash, {}):
                        p_map.setdefault(h, [0, 0.0, -np.inf])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        p_map[h][2] = max(p_map[h][2], q_log_prob)
                        pred_samples.append((h, q_log_prob))

                if not len(p_map):
                    print (f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok

                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

                # Majority voting metrics (best-of-N with log-probability ranking)
                if len(pred_samples):
                    logps = np.array([lp for _, lp in pred_samples], dtype=np.float64)
                    max_logp = logps.max()
                    probs = np.exp(logps - max_logp)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs /= prob_sum
                    else:
                        probs = np.full_like(probs, 1.0 / len(probs))

                    for sample_size in self.maj_sample_sizes:
                        sampled_indices = np.random.choice(len(pred_samples), size=sample_size, replace=True, p=probs)
                        sampled_logps = logps[sampled_indices]
                        best_idx = sampled_indices[np.argmax(sampled_logps)]
                        maj_test_correct[sample_size] += pred_samples[best_idx][0] == label_hash

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])
            for sample_size in self.maj_sample_sizes:
                maj_correct[sample_size] += maj_test_correct[sample_size] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result
        result = {f"{self.__class__.__name__}/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}
        result.update({f"{self.__class__.__name__}/maj@{k}": maj_correct[k] / len(self.test_puzzles) for k in self.maj_sample_sizes})
        return result
