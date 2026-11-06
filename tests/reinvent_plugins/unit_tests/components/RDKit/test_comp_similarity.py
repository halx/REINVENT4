import pytest

import numpy as np

from reinvent_plugins.components.RDKit.comp_similarity import (
    Parameters,
    TanimotoSimilarity,
)


@pytest.mark.parametrize(
    "smiles, radius, use_counts, use_features, expected_results",
    [
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            False,
            False,
            [1.0, 1.0],  # max aggregation: each query matches itself as a reference
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            True,
            False,
            [1.0, 1.0],
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            False,
            True,
            [1.0, 1.0],
        ),
        (
            ["c1ccccc1", "Cc1ccccc1"],
            2,
            True,
            True,
            [1.0, 1.0],
        ),
    ],
)
def test_comp_similarity(smiles, radius, use_counts, use_features, expected_results):
    params = Parameters(
        [smiles],
        [radius],
        [use_counts],
        [use_features],
    )
    td = TanimotoSimilarity(params)
    results = np.concatenate(td(smiles).scores)
    assert np.allclose(results, expected_results)
