import pytest

import numpy as np

from reinvent_plugins.components.RDKit.comp_path_similarity import (
    Parameters,
    PathFPSimilarity,
)


@pytest.mark.parametrize(
    "smiles, min_path, max_path, n_bits, expected_results",
    [
        (
            ["c1ccccc1", "Cc1ccccc1"],
            1,
            7,
            2048,
            [1.0, 1.0],  # max aggregation: each query matches itself as a reference
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            1,
            4,
            1024,
            [1.0, 1.0],
        ),
    ],
)
def test_path_fp_similarity_self(smiles, min_path, max_path, n_bits, expected_results):
    params = Parameters(
        smiles=[smiles],
        min_path=[min_path],
        max_path=[max_path],
        n_bits=[n_bits],
    )
    component = PathFPSimilarity(params)
    results = np.concatenate(component(smiles).scores)
    assert np.allclose(results, expected_results)


def test_path_fp_similarity_different_molecules():
    """Two structurally distinct molecules score strictly below 1.0 against a reference."""
    ref = ["c1ccccc1"]  # benzene
    query = ["CC(=O)O"]  # acetic acid
    params = Parameters(smiles=[ref], min_path=[1], max_path=[7], n_bits=[2048])
    component = PathFPSimilarity(params)
    results = component(query).scores[0]
    assert results[0] < 1.0
    assert results[0] >= 0.0


def test_path_fp_similarity_invalid_query():
    """Invalid query SMILES scores 0.0 rather than crashing."""
    ref = ["c1ccccc1"]
    params = Parameters(smiles=[ref], min_path=[1], max_path=[7], n_bits=[2048])
    component = PathFPSimilarity(params)
    results = component(["INVALID_SMILES"]).scores[0]
    assert results[0] == pytest.approx(0.0)


def test_path_fp_similarity_invalid_reference_raises():
    """All-invalid reference SMILES raises ValueError at construction time."""
    with pytest.raises(ValueError, match="unable to convert"):
        Parameters_ = Parameters(
            smiles=[["NOT_A_MOLECULE"]],
            min_path=[1],
            max_path=[7],
            n_bits=[2048],
        )
        PathFPSimilarity(Parameters_)


@pytest.mark.parametrize("aggregation", ["mean", "min", "max"])
def test_path_fp_similarity_aggregations(aggregation):
    """All aggregation modes return a score array of the correct length."""
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    params = Parameters(
        smiles=[smiles],
        min_path=[1],
        max_path=[7],
        n_bits=[2048],
        aggregation=[aggregation],
    )
    component = PathFPSimilarity(params)
    results = component(smiles).scores[0]
    assert len(results) == len(smiles)
    assert np.all(results >= 0.0)
    assert np.all(results <= 1.0)
