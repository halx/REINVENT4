"""Unit tests for the Loss Modulation machinery."""

import numpy as np
import pytest
import torch

from reinvent.runmodes.RL.memories.inception import Inception
from reinvent.runmodes.RL.relevance import aggregate_relevance
from reinvent.runmodes.RL.reward import RLReward
from reinvent.scoring.results import ScoreResults, TransformResults
from reinvent_plugins.components.component_results import ComponentResults


def _transform_result(relevance):
    """Minimal TransformResults carrying only what aggregate_relevance needs."""
    return TransformResults(
        component_type="dummy",
        component_names=["dummy"],
        transform_types=[None],
        transformed_scores=[np.zeros(1)],
        component_result=ComponentResults(scores=[np.zeros(1)]),
        relevance=relevance,
    )


def _results(*relevances):
    components = [_transform_result(c) for c in relevances]
    return ScoreResults(smilies=[], total_scores=np.zeros(1), completed_components=components)


def test_aggregate_relevance_none_when_absent():
    assert aggregate_relevance(_results(None, [None])) is None


def test_aggregate_relevance_arithmetic_and_min():
    results = _results([np.array([0.2, 0.8])], [np.array([0.6, 0.4])])

    np.testing.assert_allclose(aggregate_relevance(results, "arithmetic"), np.array([0.4, 0.6]))
    np.testing.assert_allclose(aggregate_relevance(results, "min"), np.array([0.2, 0.4]))


def test_aggregate_relevance_ignores_nan_rows():
    results = _results([np.array([np.nan, 0.8])], [np.array([0.6, 0.4])])

    np.testing.assert_allclose(aggregate_relevance(results, "arithmetic"), np.array([0.6, 0.6]))


def test_modulate_is_mean_one_normalised():
    loss = torch.ones(3)
    weights = torch.tensor([0.5, 1.0, 1.5])

    modulated = RLReward._modulate(loss, weights)

    # mean-1 normalisation: weights / mean(weights) = [0.5, 1.0, 1.5]
    np.testing.assert_allclose(modulated.numpy(), np.array([0.5, 1.0, 1.5]), atol=1e-6)
    assert modulated.mean().item() == pytest.approx(1.0)


def test_modulate_leaves_nan_weights_neutral():
    loss = torch.tensor([2.0, 4.0])
    weights = torch.tensor([1.0, float("nan")])

    modulated = RLReward._modulate(loss, weights)

    # first molecule scaled by 1.0/mean(finite)=1.0; second stays neutral (*1)
    np.testing.assert_allclose(modulated.numpy(), np.array([2.0, 4.0]), atol=1e-6)


def test_inception_relevance_gate_blocks_low_relevance():
    smilies = np.array(["CCO", "CCN", "CCC"])
    scores = torch.tensor([0.9, 0.8, 0.7])
    lls = torch.tensor([-1.0, -2.0, -3.0])
    relevance = np.array([0.9, 0.5, 0.7])  # middle one below the 0.6 gate

    inception = Inception(
        memory_size=10,
        sample_size=3,
        seed_smilies=None,
        scoring_function=None,
        prior=None,
        relevance_threshold=0.6,
    )
    inception.add(smilies, scores, lls, relevance=relevance)

    stored = {row[0] for row in inception.storage}
    assert stored == {"CCO", "CCC"}  # "CCN" gated out


def test_inception_admits_all_without_relevance():
    smilies = np.array(["CCO", "CCN"])
    scores = torch.tensor([0.9, 0.8])
    lls = torch.tensor([-1.0, -2.0])

    inception = Inception(
        memory_size=10,
        sample_size=2,
        seed_smilies=None,
        scoring_function=None,
        prior=None,
        relevance_threshold=0.6,
    )
    inception.add(smilies, scores, lls)  # no relevance -> gate is a no-op

    assert len(inception.storage) == 2


class _RelevanceComponent:
    number_of_endpoints = 1

    def __call__(self, smiles):
        n = len(smiles)
        return ComponentResults([np.arange(1.0, n + 1.0)], relevance=[np.linspace(0.3, 0.9, n)])


def test_compute_transform_extracts_relevance_when_enabled():
    from collections import defaultdict
    from reinvent.scoring.compute_scores import compute_transform

    smilies = ["CCO", "CCN"]
    valid_mask = np.array([True, True])
    params = (["fake"], _RelevanceComponent(), [None], [1.0], [True])

    result = compute_transform("fake", params, smilies, defaultdict(dict), valid_mask)

    assert result.relevance is not None
    np.testing.assert_allclose(result.relevance[0], np.array([0.3, 0.9]))


def test_compute_transform_ignores_relevance_when_flag_off():
    from collections import defaultdict
    from reinvent.scoring.compute_scores import compute_transform

    smilies = ["CCO", "CCN"]
    valid_mask = np.array([True, True])
    params = (["fake"], _RelevanceComponent(), [None], [1.0], [False])

    result = compute_transform("fake", params, smilies, defaultdict(dict), valid_mask)

    assert result.relevance is None


class _EmptyInception:
    """Stub inception whose memory is still empty, so it replays nothing."""

    def __call__(self, *args, **kwargs):
        return None


def test_reward_tolerates_empty_inception():
    from reinvent.runmodes.RL.reward import dap_strategy

    param = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.SGD([param], lr=0.1)
    reward = RLReward(optimizer, sigma=1.0, strategy=dap_strategy)

    scores = np.array([0.5, 0.6, 0.7])
    agent_nlls = param + 1.0  # keeps a grad path so loss.backward() works
    prior_nlls = torch.ones(3)
    mask_idx = np.array([0, 1, 2])

    # inception returns None (empty buffer) -> must not raise
    _, _, _, loss = reward(
        ["C", "CC", "CCC"],
        scores,
        agent_nlls,
        prior_nlls,
        mask_idx,
        _EmptyInception(),
        agent=None,
        weights=np.array([0.5, 1.0, 1.5]),
    )

    assert torch.isfinite(loss)
