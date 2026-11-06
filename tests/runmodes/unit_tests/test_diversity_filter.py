import pytest

import numpy as np
import torch

from reinvent.runmodes.RL.memories.filters import IdenticalMurckoScaffold
from reinvent.runmodes.RL.memories.penalties import StepPenalty
from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState


def _make_batch(smilies, states=None):
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
def diversity_filter():
    return IdenticalMurckoScaffold(
        penalty_function=StepPenalty(2),
        intrinsic_reward=None,
        bucket_size=2,
        minscore=0.4,
        minsimilarity=0.4,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
    )


def test_filter_updates_scaffold_memory(diversity_filter):
    scores = np.array([0.64, 0.55, 0.897, 0.737])
    smilies = ["CCc1ccccc1", "CCC=O=O", "CCCCc1ccccc1", "CCc1cnccc1"]
    states = np.array(
        [SmilesState.VALID, SmilesState.INVALID, SmilesState.VALID, SmilesState.DUPLICATE]
    )
    mask = np.where((states == SmilesState.VALID) | (states == SmilesState.DUPLICATE), True, False)

    diversity_filter.update_score(scores, _make_batch(smilies, states), mask)

    assert len(diversity_filter.scaffold_memory) == 2


def test_filter_invalid_smiles_not_stored(diversity_filter):
    scores = np.array([0.9, 0.1])
    smilies = ["CCC=O=O", "CCC=O=O"]
    states = np.array([SmilesState.INVALID, SmilesState.INVALID])
    mask = np.where((states == SmilesState.VALID) | (states == SmilesState.DUPLICATE), True, False)

    diversity_filter.update_score(scores, _make_batch(smilies, states), mask)

    assert len(diversity_filter.scaffold_memory) == 0


def test_filter_second_call_same_scaffold_scores_zero(diversity_filter):
    """SMILES already in smiles_memory should have their score zeroed on the next call."""
    scores_1 = np.array([0.8, 0.9])
    smilies_1 = ["CCc1ccccc1", "CCc1ccccc1"]
    mask_1 = np.array([True, True])
    diversity_filter.update_score(scores_1, _make_batch(smilies_1), mask_1)

    scores_2 = np.array([0.7])
    smilies_2 = ["CCc1ccccc1"]
    mask_2 = np.array([True])
    diversity_filter.update_score(scores_2, _make_batch(smilies_2), mask_2)

    assert scores_2[0] == 0.0
