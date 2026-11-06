"""Loss modulation: aggregate per-endpoint relevance into a per-molecule relevance weight.

Implements the aggregation step of the Loss Modulation strategy: each endpoint that opted in
provides a relevance in [0,1] (1 = fully relevant, 0 = ignore this molecule); these are combined
across endpoints into a single per-molecule weight.  The mean-1 normalisation of
Eq. (LossModulationLoss) is applied later, in RLReward, so it can span the sampled batch and the
inception replay together.
"""

from __future__ import annotations

__all__ = ["AGGREGATORS", "aggregate_relevance"]

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

# NaN entries mark molecules without relevance (e.g. invalid/duplicate); they stay neutral later.
AGGREGATORS = {
    "arithmetic": lambda m: np.nanmean(m, axis=0),
    "min": lambda m: np.nanmin(m, axis=0),
}


def aggregate_relevance(
    results: ScoreResults, aggregation: str = "arithmetic"
) -> Optional[np.ndarray]:
    """Aggregate the relevance of all loss-modulation endpoints into one weight per molecule.

    :param results: scoring results holding per-endpoint relevance on completed components
    :param aggregation: how to combine endpoints, one of AGGREGATORS
    :returns: per-molecule relevance array (NaN where no endpoint provided one), or None
    """
    cols = [
        np.asarray(c, dtype=float)
        for tr in results.completed_components
        if tr.relevance is not None
        for c in tr.relevance
        if c is not None
    ]

    if not cols:
        return None

    stacked = np.vstack(cols)

    with np.errstate(
        invalid="ignore"
    ):  # ignore warnings from all-NaN columns when no relevance is provided
        return AGGREGATORS[aggregation](stacked)
