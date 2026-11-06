"""Regression test for the NaN-score / valid-mask index misalignment in ``RLReward``.

``mask_idx`` lives in the *original* batch index space (the same space as
``orig_smilies``), but ``scores_nonnan`` and ``prior_lls`` were produced by
filtering with ``~nan_idx``.  When a molecule with a NaN score appears *before*
one or more valid molecules, indexing the ``~nan_idx``-filtered tensors with the
original-space ``mask_idx`` selects the wrong rows.  As a result inception is
handed SMILES paired with the scores/prior-lls of *different* molecules.

This test drives ``RLReward.__call__`` with a batch whose first molecule has a
NaN score and asserts that inception receives each valid SMILES paired with its
own score.  It fails if ``reward.py`` indexes the ``~nan_idx``-filtered tensors
with the original-space ``mask_idx`` instead of intersecting the valid and
non-NaN masks in the original batch space.
"""

import numpy as np
import torch
from torch import nn

from reinvent.runmodes.RL.reward import RLReward, dap_strategy


class _CapturingInception:
    """Record what ``RLReward`` passes to inception, then echo it back unchanged."""

    def __init__(self):
        self.smilies = None
        self.scores = None

    def __call__(self, smilies, scores, prior_lls, agent_lls=None, agent=None, relevance=None):
        self.smilies = [str(s) for s in np.asarray(smilies)]
        self.scores = [float(x) for x in scores]
        return smilies, scores, prior_lls, agent_lls, relevance


class _ZeroAgent:
    """Minimal agent stub taking the Reinvent (tensor) branch of ``likelihood_smiles``."""

    def likelihood_smiles(self, smilies):
        return torch.zeros(len(smilies), requires_grad=True)


def test_inception_receives_aligned_scores_when_nan_present():
    param = nn.Parameter(torch.zeros(4))
    reward = RLReward(torch.optim.SGD([param], lr=0.0), sigma=1.0, strategy=dap_strategy)

    orig_smilies = ["A", "B", "C", "D"]
    # "A" failed scoring (NaN); "B", "C", "D" scored fine.
    scores = np.array([np.nan, 1.0, 2.0, 3.0])
    # "A", "B", "C" are the valid SMILES fed to inception; "D" is excluded
    # (e.g. a duplicate) even though it scored.
    mask_idx = np.array([0, 1, 2])

    inception = _CapturingInception()
    reward(
        orig_smilies,
        scores,
        param,  # agent_nlls
        torch.zeros(4),  # prior_nlls
        mask_idx,
        inception,
        _ZeroAgent(),
    )

    received = dict(zip(inception.smilies, inception.scores))

    # "B" is a valid SMILES whose true score is 1.0, so inception must receive
    # "B" paired with 1.0.  With the bug it is paired with 2.0 (the score that
    # belongs to "C") because the NaN "A" shifts the filtered arrays.
    assert received.get("B") == 1.0, f"inception received misaligned scores: {received}"
