from __future__ import annotations

import numpy as np

from ..diversity_filter import DiversityFilter


class PenalizeSameSmiles(DiversityFilter):
    """Penalize previously generated compounds."""

    def calculate_penalty(
        self, scores: np.ndarray, smilies: list[str]
    ) -> tuple[np.ndarray, None | list[tuple[list[int], int]]]:
        penalties = np.ones_like(scores)
        if self.penalty_multiplier is not None:
            penalties *= self.penalty_multiplier
        return penalties, None

    def peek_penalty(self, scores: np.ndarray, smilies: list[str]) -> np.ndarray:
        """Return penalties without mutating state (identical to calculate_penalty here).

        :param scores: scores array — used only for shape and dtype
        :param smilies: SMILES strings to query
        :returns: penalty multipliers, one per entry in *smilies*
        """
        penalties = np.ones_like(scores)
        if self.penalty_multiplier is not None:
            penalties *= self.penalty_multiplier
        return penalties

    def count_full_buckets(self) -> int:
        return 0

    def count_total_buckets(self) -> int:
        return 0
