"""Extended tests for TanimotoSimilarity — multi-reference and edge cases."""

import numpy as np
import pytest

from reinvent_plugins.components.RDKit.comp_similarity import TanimotoSimilarity, Parameters
from tests.test_data import ASPIRIN, CELECOXIB, BENZENE


def _make_params(smiles_list, radius=3, use_counts=True, use_features=True):
    return Parameters(
        smiles=[smiles_list],
        radius=[radius],
        use_counts=[use_counts],
        use_features=[use_features],
    )


# ── Single reference ──────────────────────────────────────────────────────────


def test_similarity_to_self_is_one():
    params = _make_params([ASPIRIN])
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN])
    assert results.scores[0][0] == pytest.approx(1.0, abs=1e-4)


def test_similarity_to_unrelated_is_low():
    params = _make_params([ASPIRIN])
    comp = TanimotoSimilarity(params)
    results = comp([BENZENE])
    assert results.scores[0][0] < 0.5


# ── Multiple references ───────────────────────────────────────────────────────


def test_similarity_multiple_references_returns_two_score_arrays():
    """Two reference SMILES → two endpoints → two score arrays."""
    params = Parameters(
        smiles=[[ASPIRIN], [CELECOXIB]],
        radius=[3, 3],
        use_counts=[True, True],
        use_features=[True, True],
    )
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN, CELECOXIB])
    assert len(results.scores) == 2


def test_similarity_perfect_match_in_multi_reference():
    params = Parameters(
        smiles=[[ASPIRIN], [CELECOXIB]],
        radius=[3, 3],
        use_counts=[True, True],
        use_features=[True, True],
    )
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN])
    # first endpoint (vs ASPIRIN): should be 1.0
    assert results.scores[0][0] == pytest.approx(1.0, abs=1e-4)
    # second endpoint (vs CELECOXIB): should be low
    assert results.scores[1][0] < 0.5


# ── Invalid reference raises ──────────────────────────────────────────────────


def test_invalid_reference_smiles_raises():
    params = _make_params(["INVALID_SMILES"])
    with pytest.raises((ValueError, Exception)):
        TanimotoSimilarity(params)


# ── Output shape ──────────────────────────────────────────────────────────────


def test_similarity_output_length_matches_input():
    params = _make_params([ASPIRIN])
    comp = TanimotoSimilarity(params)
    query = [ASPIRIN, CELECOXIB, BENZENE]
    results = comp(query)
    assert len(results.scores[0]) == len(query)


# ── Aggregation parameter ─────────────────────────────────────────────────────


def _make_params_with_aggregation(smiles_list, aggregation, radius=3, use_counts=True, use_features=True):
    return Parameters(
        smiles=[smiles_list],
        radius=[radius],
        use_counts=[use_counts],
        use_features=[use_features],
        aggregation=[aggregation],
    )


def test_aggregation_default_is_max():
    """When aggregation is not specified, default should behave like max."""
    params_default = _make_params([ASPIRIN, CELECOXIB])
    params_max = _make_params_with_aggregation([ASPIRIN, CELECOXIB], "max")
    comp_default = TanimotoSimilarity(params_default)
    comp_max = TanimotoSimilarity(params_max)
    query = [ASPIRIN, BENZENE]
    results_default = comp_default(query)
    results_max = comp_max(query)
    assert np.allclose(results_default.scores[0], results_max.scores[0])


def test_aggregation_max():
    """Max aggregation should return the maximum score across references."""
    params = _make_params_with_aggregation([ASPIRIN, BENZENE], "max")
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN])
    # ASPIRIN vs itself is 1.0, BENZENE vs ASPIRIN is < 1.0, max should be 1.0
    assert results.scores[0][0] == pytest.approx(1.0, abs=1e-4)


def test_aggregation_min():
    """Min aggregation should return the minimum score across references."""
    params = _make_params_with_aggregation([ASPIRIN, BENZENE], "min")
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN])
    # ASPIRIN vs itself is 1.0, BENZENE vs ASPIRIN is < 1.0, min should be < 1.0
    assert results.scores[0][0] < 1.0


def test_aggregation_mean():
    """Mean aggregation should return the average score across references."""
    params = _make_params_with_aggregation([ASPIRIN, BENZENE], "mean")
    comp = TanimotoSimilarity(params)
    results = comp([ASPIRIN])
    # mean of (1.0, <1.0) should be between min and max
    assert results.scores[0][0] < 1.0
    assert results.scores[0][0] > 0.0


def test_aggregation_produces_single_score_array():
    """Aggregation should produce one score array per endpoint, not per reference."""
    params = _make_params_with_aggregation([ASPIRIN, CELECOXIB, BENZENE], "mean")
    comp = TanimotoSimilarity(params)
    query = [ASPIRIN, CELECOXIB]
    results = comp(query)
    assert len(results.scores) == 1
    assert len(results.scores[0]) == 2
