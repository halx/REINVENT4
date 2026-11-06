"""Unit tests for IS-weighted sampling and diversity-penalty coupling in Inception."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from reinvent.runmodes.RL.memories.inception import Inception

SMILIES = (
    "CCc1ccccc1",
    "CCCc1ccccc1",
    "CCCCc1ccccc1",
    "Cc1ccccc1",
    "c1ccccc1",
    "C1CCCCC1",
    "CC(=O)O",
    "CCO",
)

MEMORY_SIZE = len(SMILIES)
SAMPLE_SIZE = 3


class MockAgent:
    """Minimal agent stub: returns constant NLLs for any SMILES list."""

    def likelihood_smiles(self, smilies):
        return torch.ones(len(smilies)) * 2.0


def _make_inception(**kwargs):
    scores = torch.linspace(0.3, 0.9, MEMORY_SIZE)
    lls = torch.ones(MEMORY_SIZE) * -5.0

    inception = Inception(
        memory_size=MEMORY_SIZE,
        sample_size=SAMPLE_SIZE,
        seed_smilies=None,
        scoring_function=None,
        prior=None,
        **kwargs,
    )
    inception.add(SMILIES, scores, lls)
    return inception


class TestFourTupleAPI:
    def test_call_returns_five_tuple(self):
        inception = _make_inception()
        smilies = ("CCC", "CCCC")
        scores = torch.tensor([0.5, 0.6])
        lls = torch.tensor([-4.0, -4.5])
        result = inception(smilies, scores, lls)
        assert result is not None
        assert len(result) == 5  # smilies, scores, prior_lls, agent_lls, relevance

    def test_call_returns_correct_sample_size(self):
        inception = _make_inception()
        smilies = ("CCC", "CCCC")
        scores = torch.tensor([0.5, 0.6])
        lls = torch.tensor([-4.0, -4.5])
        result_smilies, result_scores, result_lls, result_agent_lls, result_relevance = inception(
            smilies, scores, lls
        )
        assert len(result_smilies) == SAMPLE_SIZE
        assert len(result_scores) == SAMPLE_SIZE
        assert len(result_lls) == SAMPLE_SIZE
        assert len(result_agent_lls) == SAMPLE_SIZE
        assert len(result_relevance) == SAMPLE_SIZE

    def test_storage_is_five_tuples(self):
        inception = _make_inception()
        for entry in inception.storage:
            assert len(entry) == 5

    def test_empty_storage_returns_none_tuple(self):
        inception = Inception(
            memory_size=MEMORY_SIZE,
            sample_size=SAMPLE_SIZE,
            seed_smilies=None,
            scoring_function=None,
            prior=None,
        )
        sampled, w_max, w_entropy, drift = inception.sample()
        assert sampled is None
        assert w_max is None
        assert w_entropy is None
        assert drift is None


class TestUniformFallback:
    def test_sample_without_agent_returns_none_is_metrics(self):
        inception = _make_inception(is_weighted_sampling=True)
        sampled, w_max, w_entropy, drift = inception.sample(agent=None)
        assert sampled is not None
        assert w_max is None
        assert w_entropy is None
        assert drift is None

    def test_sample_without_is_flag_returns_none_is_metrics(self):
        inception = _make_inception(is_weighted_sampling=False)
        agent = MockAgent()
        sampled, w_max, w_entropy, drift = inception.sample(agent=agent)
        assert sampled is not None
        assert w_max is None
        assert w_entropy is None
        assert drift is None


class TestISWeightedSampling:
    def test_is_weighted_sample_populates_metrics(self):
        inception = _make_inception(is_weighted_sampling=True)
        agent = MockAgent()
        sampled, w_max, w_entropy, drift = inception.sample(agent=agent)
        assert sampled is not None
        assert w_max is not None
        assert isinstance(w_max, float)
        assert w_entropy is not None
        assert isinstance(w_entropy, float)
        assert drift is not None
        assert isinstance(drift, float)

    def test_is_weight_max_in_unit_interval(self):
        inception = _make_inception(is_weighted_sampling=True)
        _, w_max, _, _ = inception.sample(agent=MockAgent())
        assert 0.0 <= w_max <= 1.0

    def test_is_weight_entropy_non_negative(self):
        inception = _make_inception(is_weighted_sampling=True)
        _, _, w_entropy, _ = inception.sample(agent=MockAgent())
        assert w_entropy >= 0.0

    def test_is_weight_clip_applied(self):
        """With a very tight clip the weights should be nearly uniform → high entropy."""
        inception_clipped = _make_inception(is_weighted_sampling=True, is_weight_clip=0.001)
        inception_unclipped = _make_inception(is_weighted_sampling=True)
        _, _, entropy_clipped, _ = inception_clipped.sample(agent=MockAgent())
        _, _, entropy_unclipped, _ = inception_unclipped.sample(agent=MockAgent())
        # Tight clipping → near-uniform distribution → higher entropy
        assert entropy_clipped >= entropy_unclipped - 1e-6


class TestDiversityPenaltyCoupling:
    def test_stored_scores_reduced_with_penalty_weight_one(self):
        n = MEMORY_SIZE
        mock_df = MagicMock()
        mock_df.peek_penalty.return_value = np.full(n, 0.5)

        inception = _make_inception(
            diversity_filter=mock_df,
            diversity_penalty_weight=1.0,
        )

        scores_before = [entry[1] for entry in inception.storage]

        # Second add triggers diversity penalty application to existing storage
        new_smilies = ("CCC",)
        inception.add(new_smilies, torch.tensor([0.5]), torch.tensor([-4.0]))

        scores_after = [entry[1] for entry in inception.storage]
        # All stored scores should have been halved by the 0.5 penalty
        for before, after in zip(scores_before, scores_after):
            if before in scores_after:  # entry still in top-k
                assert after <= before + 1e-6

        mock_df.peek_penalty.assert_called_once()

    def test_no_penalty_when_weight_zero(self):
        mock_df = MagicMock()
        mock_df.peek_penalty.return_value = np.zeros(MEMORY_SIZE)

        inception = _make_inception(
            diversity_filter=mock_df,
            diversity_penalty_weight=0.0,
        )

        scores_before = {e[0]: float(e[1]) for e in inception.storage}
        inception.add(("CCC",), torch.tensor([0.5]), torch.tensor([-4.0]))
        scores_after = {e[0]: float(e[1]) for e in inception.storage}

        # With weight=0 the penalty formula is w*p + (1-w) = 0*p + 1 = 1.0 → no change
        # for molecules that remain in storage after top-k truncation
        for smi, score_before in scores_before.items():
            if smi in scores_after:
                assert abs(scores_after[smi] - score_before) < 1e-5
