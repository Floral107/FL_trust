# TensorFlow-compatible GTGShapley implementation

import copy
import math
import numpy as np
# tensorflow imported lazily — the class itself only needs numpy, so we
# avoid forcing TF/CUDA initialization on machines that don't have a working
# TF (e.g. the dev laptop). If a downstream caller needs tf, they import it
# themselves.
try:
    import tensorflow as tf  # noqa: F401  (kept for backward compat with callers)
except Exception:
    tf = None  # type: ignore[assignment]
from typing import List, Dict, Tuple, Callable, Optional


class GTGShapley:
    def __init__(
        self,
        num_players: int,
        last_round_utility: float = 0.0,
        eps: float = 0.001,
        round_trunc_threshold: float = 0.002,
        convergence_criteria: float = 0.05,
        last_k: int = 10,
        converge_min: int = 30,
        max_percentage: float = 0.5,
        prefix_length: int = 1,
        normalize: bool = False,
        loss_flag: bool = False,
        # Hard cap on permutations. Without this, methods whose true Shapley
        # values are near zero (rob, adv_pgd on CelebA — adding/removing one
        # client barely moves the metric) never trigger relative-error
        # convergence and the loop spins until max_number (~0.5M permutations).
        # 30 permutations × ~20 subsets/perm × dedup is the standard GTG-Shapley
        # configuration (Liu et al. 2022) and gives ±0.02 Shapley estimate noise
        # for our [0,1]-valued metrics — well within the round-to-round variance
        # of FL training itself. Earlier we used 100 as a safety cap but the
        # PGD-heavy methods (gtg_adv_pgd) take ~6 hr/round at that setting,
        # which is unworkable on the GPU budget.
        max_permutations: int = 30,
        # Absolute floor for the convergence denominator. When |last| < this,
        # we treat the convergence check as absolute (|recent - last| <= criterion)
        # instead of relative — otherwise tiny Shapley values give huge relative
        # errors and prevent convergence. 0.01 ≈ scale of meaningful Shapley
        # contributions for our metrics (which live in [0, 1]).
        convergence_denom_floor: float = 0.01,
    ):
        self.num_players = num_players
        self.last_round_utility = last_round_utility
        self.eps = eps
        self.round_trunc_threshold = round_trunc_threshold
        self.convergence_criteria = convergence_criteria
        self.last_k = min(last_k, num_players)
        self.converge_min = max(converge_min, num_players)
        self.max_number = min(
            2**num_players,
            max(self.converge_min, int(max_percentage * (2**num_players)) + np.random.randint(-5, 5)),
            max_permutations,  # hard cap — see docstring above
        )
        self.convergence_denom_floor = convergence_denom_floor
        self.prefix_length = prefix_length
        self.normalize = normalize
        self.shapley_values = {}
        self.shapley_values_best_subset = {}
        self.utility_function = None
        self.evaluated_subsets = {}
        self.loss_flag = loss_flag


    def set_utility_function(self, utility_function: Callable):
        self.utility_function = utility_function

    def evaluate_subset(self, subset: List[int]) -> float:
        if not subset:
            return float(self.utility_function([]))
        subset_key = tuple(sorted(subset))
        if subset_key in self.evaluated_subsets:
            return float(self.evaluated_subsets[subset_key])
        utility = float(self.utility_function(subset))
        self.evaluated_subsets[subset_key] = utility
        return utility

    def compute(self, round_num: int = 0, return_raw: bool = False) -> List[float]:
        """Compute per-player Shapley values for the current round.

        Args:
            round_num: round identifier (kept for backwards compat; not used internally).
            return_raw: if False (default — existing behavior), apply the
                "best subset" truncation: clients outside the highest-utility
                coalition receive 0. This is what robustness.py / cont_evals.py
                expect and what produced data/combo/*.csv.
                If True, return the *raw* mean-marginal-contribution Shapley
                array (every client gets a value, including negative). Used by
                ST/DY in-loop scoring where we want to rank all clients rather
                than zero out non-best-subset ones.

        Returns:
            list of length num_players with per-player Shapley estimates.
        """
        self.shapley_values = {}
        self.shapley_values_best_subset = {}
        self.evaluated_subsets = {}
        all_players = list(range(self.num_players))
        current_utility = float(self.evaluate_subset(all_players))
        if abs(current_utility - self.last_round_utility) <= self.round_trunc_threshold:
            self.last_round_utility = current_utility
            return [0.0 for _ in all_players]

        contribution_records = []
        permutation_index = 0

        while not self._check_convergence(permutation_index, contribution_records):
            start_player = permutation_index % self.num_players
            prefix = [(start_player + i) % self.num_players for i in range(self.prefix_length)]
            prefix_set = set(prefix)
            remaining_players = [p for p in all_players if p not in prefix_set]
            np.random.shuffle(remaining_players)
            perm = prefix + remaining_players

            empty_utility = float(self.evaluate_subset([]))
            utilities = [empty_utility]
            marginal_contributions = [0.0] * self.num_players

            for j in range(self.num_players):
                current_player = perm[j]
                subset = perm[:j+1]
                if abs(current_utility - utilities[-1]) < self.eps:
                    for rp in perm[j:]:
                        marginal_contributions[rp] = 0.0
                    break
                subset_utility = float(self.evaluate_subset(subset))
                utilities.append(subset_utility)
                marginal_contributions[current_player] = float(utilities[-1] - utilities[-2])

            contribution_records.append(marginal_contributions)
            permutation_index += 1

        shapley_array = np.mean(contribution_records, axis=0)
        self.last_round_utility = current_utility
        overall_raw_shapley = {i: float(shapley_array[i]) for i in range(self.num_players)}

        if return_raw:
            # Ranking-friendly: every client gets a value. Used by ST/DY where
            # we want to weight all clients by their relative contribution.
            return [overall_raw_shapley[i] for i in range(self.num_players)]

        # Default: best-subset truncation (existing behavior).
        best_subset = self._find_best_subset()
        overall_marginal_gain = current_utility - self.evaluate_subset([])
        final_values = self._calculate_best_subset_values(best_subset, overall_raw_shapley, overall_marginal_gain)
        return [final_values[i] for i in range(self.num_players)]


    def _check_convergence(self, index: int, records: List) -> bool:
        if index < self.converge_min:
            return False
        if index >= self.max_number:
            return True
        if len(records) < self.last_k:
            return False
        all_values = (
            np.cumsum(records, axis=0) /
            np.reshape(np.arange(1, len(records)+1), (-1, 1))
        )
        recent = all_values[-self.last_k:]
        last = all_values[-1:]
        # Hybrid convergence: relative error when |last| is large enough to be
        # meaningful; absolute floor otherwise. The old `(|last| + 1e-12)`
        # denominator produced huge relative errors when Shapley values were
        # near zero (e.g. rob/adv methods where one client barely moves the
        # metric), preventing convergence entirely — the loop then ran out to
        # max_number permutations (effectively forever).
        denom = np.maximum(np.abs(last), self.convergence_denom_floor)
        errors = np.mean(np.abs(recent - last) / denom, axis=1)
        return np.max(errors) <= self.convergence_criteria

    def _find_best_subset(self) -> List[int]:
        if not self.evaluated_subsets:
            return []
        valid = {k: v for k, v in self.evaluated_subsets.items() if k}
        if not valid:
            return []

        if self.loss_flag:
            # loss: min + legr�videbb
            best = min(valid.items(), key=lambda x: (x[1], len(x[0])))
        else:
            # utility: max + legr�videbb
            best = max(valid.items(), key=lambda x: (x[1], -len(x[0])))

        return list(best[0])


    def _calculate_best_subset_values(
        self,
        best_subset: List[int],
        overall_raw_shapley: Dict[int, float],
        overall_marginal_gain: float
    ) -> Dict[int, float]:
        if not best_subset:
            return {i: 0.0 for i in range(self.num_players)}
        best_set = set(best_subset)
        raw_values = {
            i: (overall_raw_shapley.get(i, 0.0) if i in best_set else 0.0)
            for i in range(self.num_players)
        }
        if self.normalize:
            return self._normalize_values(raw_values, overall_marginal_gain)
        return raw_values

    def _normalize_values(self, values: Dict[int, float], marginal_gain: float) -> Dict[int, float]:
        """
        Signed min-shift normalization:
        --------------------------------
        - Stabilizes Shapley values by shifting all by |min|.
        - Keeps the original sign of each player's contribution.
        - Normalizes magnitudes so total absolute sum = 1.
        - Produces stable, comparable results across rounds.
    
        Args:
            values: Raw Shapley values {player_id: value}.
            marginal_gain: Total utility difference (unused but kept for consistency).
    
        Returns:
            Dict[int, float]: Normalized (signed) Shapley values.
        """
    
        num_players = self.num_players
        vals = [values.get(i, 0.0) for i in range(num_players)]
    
        # Step 0: handle degenerate case (all zero or near-zero)
        if all(abs(v) < 1e-12 for v in vals):
            return {i: 1.0 / num_players for i in range(num_players)}
      
        print("raw_values")
        print(vals)
        # Step 1: shift all values upward by |min| to stabilize scale
        min_val = min(vals)
        shift = abs(min_val)
        shifted = [v + shift for v in vals]
    
        # Step 2: normalize by total absolute sum (avoid zero division)
        total_abs = sum(abs(v) for v in shifted)
        if total_abs < 1e-12:
            return {i: 1.0 / num_players for i in range(num_players)}
    
        # Step 3: apply sign of original values after normalization
        normalized = {
            i: math.copysign(abs(shifted[i]) / total_abs, vals[i])
            for i in range(num_players)
        }
        print("norm_values")
        print(normalized)
        return normalized


