"""Unit tests for DiversityFilter.update_score() via IdenticalMurckoScaffold."""

import numpy as np
import pytest
import torch

from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.runmodes.RL.memories.filters.identical_murcko_scaffold import IdenticalMurckoScaffold
from reinvent.runmodes.RL.memories.penalties.step_penalty import StepPenalty
from reinvent.runmodes.RL.memories.utils.diversity_results import DiversityResults

BUCKET_SIZE = 2
MINSCORE = 0.4


def make_batch(smilies, states=None):
    n = len(smilies)
    if states is None:
        states = np.array([SmilesState.VALID] * n)
    return SampleBatch(
        items1=None,
        items2=list(smilies),
        nlls=torch.zeros(n),
        smilies=list(smilies),
        states=states,
    )


@pytest.fixture
def df():
    return IdenticalMurckoScaffold(
        penalty_function=StepPenalty(BUCKET_SIZE),
        intrinsic_reward=None,
        bucket_size=BUCKET_SIZE,
        minscore=MINSCORE,
        minsimilarity=None,
        penalty_multiplier=None,
        rdkit_smiles_flags={},
    )


# Two molecules sharing the same Murcko scaffold (ethylbenzene / propylbenzene → benzene ring)
ETHYLBENZENE = "CCc1ccccc1"
PROPYLBENZENE = "CCCc1ccccc1"
BUTYLBENZENE = "CCCCc1ccccc1"
CYCLOHEXANE = "C1CCCCC1"  # different scaffold


class TestUpdateScoreAlwaysReturns:
    def test_returns_diversity_results_not_none(self, df):
        scores = np.array([0.8])
        batch = make_batch([ETHYLBENZENE])
        mask = np.array([True])
        result = df.update_score(scores, batch, mask)
        assert result is not None
        assert isinstance(result, DiversityResults)

    def test_memory_size_always_populated(self, df):
        scores = np.array([0.8])
        batch = make_batch([ETHYLBENZENE])
        mask = np.array([True])
        result = df.update_score(scores, batch, mask)
        assert result.memory_size is not None
        assert isinstance(result.memory_size, int)

    def test_returns_diversity_results_when_no_active(self, df):
        """All scores below minscore → active_idxs empty; must still return DiversityResults."""
        scores = np.array([0.1, 0.2])
        batch = make_batch([ETHYLBENZENE, PROPYLBENZENE])
        mask = np.array([True, True])
        result = df.update_score(scores, batch, mask)
        assert result is not None
        assert isinstance(result, DiversityResults)
        assert result.memory_size == 0


class TestSmilesDeduplication:
    def test_seen_smiles_scored_zero_on_second_call(self, df):
        scores1 = np.array([0.8])
        batch1 = make_batch([ETHYLBENZENE])
        mask1 = np.array([True])
        df.update_score(scores1, batch1, mask1)

        scores2 = np.array([0.9])
        batch2 = make_batch([ETHYLBENZENE])
        mask2 = np.array([True])
        df.update_score(scores2, batch2, mask2)

        assert scores2[0] == 0.0

    def test_different_smiles_same_scaffold_not_deduped_as_smiles(self, df):
        scores1 = np.array([0.8])
        df.update_score(scores1, make_batch([ETHYLBENZENE]), np.array([True]))

        # Different SMILES, same scaffold — should not be deduped by smiles memory
        scores2 = np.array([0.9])
        df.update_score(scores2, make_batch([PROPYLBENZENE]), np.array([True]))

        assert scores2[0] != 0.0  # not zeroed by dedup (may be zeroed by bucket later)


class TestMinscoreThreshold:
    def test_below_minscore_not_stored(self, df):
        scores = np.array([MINSCORE - 0.01])
        df.update_score(scores, make_batch([ETHYLBENZENE]), np.array([True]))
        assert df.get_memory_size() == 0

    def test_at_minscore_stored(self, df):
        scores = np.array([MINSCORE])
        df.update_score(scores, make_batch([ETHYLBENZENE]), np.array([True]))
        assert df.get_memory_size() == 1

    def test_above_minscore_stored(self, df):
        scores = np.array([MINSCORE + 0.1])
        df.update_score(scores, make_batch([ETHYLBENZENE]), np.array([True]))
        assert df.get_memory_size() == 1


class TestMaskRespected:
    def test_invalid_smiles_not_stored(self, df):
        scores = np.array([0.9, 0.8])
        states = np.array([SmilesState.INVALID, SmilesState.VALID])
        batch = make_batch([ETHYLBENZENE, CYCLOHEXANE], states=states)
        mask = np.where(states != SmilesState.INVALID, True, False)
        df.update_score(scores, batch, mask)
        assert df.get_memory_size() == 1  # only cyclohexane (valid)


class TestBucketZeroing:
    def test_second_molecule_in_bucket_not_zeroed(self, df):
        """With bucket_size=2, the 2nd molecule filling the bucket must NOT be zeroed."""
        scores1 = np.array([0.8, 0.9])
        batch1 = make_batch([ETHYLBENZENE, PROPYLBENZENE])
        mask1 = np.array([True, True])
        df.update_score(scores1, batch1, mask1)

        # Both should be non-zero — bucket allows bucket_size molecules
        assert scores1[0] > 0.0
        assert scores1[1] > 0.0

    def test_third_molecule_over_bucket_zeroed(self, df):
        """The 3rd molecule in a bucket of size 2 should be zeroed (bucket_size+1)."""
        # Fill the bucket with 2 molecules from the same scaffold
        scores1 = np.array([0.8, 0.9])
        df.update_score(scores1, make_batch([ETHYLBENZENE, PROPYLBENZENE]), np.array([True, True]))

        # 3rd molecule with same scaffold — should be zeroed
        scores2 = np.array([0.7])
        df.update_score(scores2, make_batch([BUTYLBENZENE]), np.array([True]))

        assert scores2[0] == pytest.approx(0.0)

    def test_different_scaffold_not_affected_by_full_bucket(self, df):
        """A molecule with a different scaffold is unaffected by another scaffold's full bucket."""
        scores1 = np.array([0.8, 0.9])
        df.update_score(scores1, make_batch([ETHYLBENZENE, PROPYLBENZENE]), np.array([True, True]))
        scores1_overflow = np.array([0.7])
        df.update_score(scores1_overflow, make_batch([BUTYLBENZENE]), np.array([True]))

        scores2 = np.array([0.85])
        df.update_score(scores2, make_batch([CYCLOHEXANE]), np.array([True]))

        assert scores2[0] > 0.0
