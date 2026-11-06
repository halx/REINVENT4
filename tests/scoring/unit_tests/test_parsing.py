import pytest

from reinvent.scoring.config import collect_params
from reinvent.scoring.scorer import get_components
from dataclasses import fields


def test_get_components():
    components = [
        {"QED": {"endpoint": [{"name": "QED drug-like score", "weight": 0.79}]}},
        {
            "MolecularWeight": {
                "endpoint": [
                    {
                        "name": "MW",
                        "transform": {
                            "type": "reverse_sigmoid",
                            "high": 1.2,
                            "low": 0.85,
                            "k": 0.2,
                        },
                        "weight": 0.79,
                    }
                ]
            }
        },
    ]

    components_dict = get_components(components)

    assert len(components_dict.scorers) == 2
    assert "qed" == components_dict.scorers[0].component_type
    assert "molecularweight" == components_dict.scorers[1].component_type
    for component in components_dict.scorers:
        assert len(fields(component)) == 3  # name, params, cache
        assert (
            len(component.params) == 5
        )  # name, component object, transform, weight, loss_modulation

    assert not components_dict.filters
    assert not components_dict.penalties


def test_complevel_params():
    components = [
        {"QED": {"endpoint": [{"name": "QED drug-like score", "weight": 0.79}]}},
        {
            "external_process": {
                "params": {  # Component-level params.
                    "executable": "path/to/executable1",
                    "args": "--loglevel DEBUG",
                },
                "endpoint": [
                    {
                        "name": "Endpoint1",
                        "weight": 0.5,
                        "params": {  # Endpoint-level params.
                            "property": "Property 1",  # Endpoint key.
                        },
                    },
                    {
                        "name": "Endpoint2",
                        "weight": 0.7,
                        "params": {  # Endpoint-level params.
                            "property": "Property 2",  # Endpoint key.
                        },
                    },
                ],
            }
        },
        {
            "external_process": {
                "params": {  # Component-level params.
                    "executable": "path/to/executable1",
                    "args": "",
                    "property": "Property 1",  # Will be overridden by endpoint.
                },
                "endpoint": [
                    {
                        "name": "Endpoint1",
                        "weight": 0.5,
                        "params": {  # Endpoint-level params.
                            # Property not overridden here, will use component-level.
                        },
                    },
                    {
                        "name": "Endpoint2",
                        "weight": 0.7,
                        "params": {  # Endpoint-level params.
                            "property": "Property 2",  # Overridden by endpoint.
                        },
                    },
                ],
            }
        },
    ]

    components_dict = get_components(components)

    assert len(components_dict.scorers) == 3
    assert "qed" == components_dict.scorers[0].component_type

    assert "externalprocess" == components_dict.scorers[1].component_type
    name, comp, transform, weights, loss_modulation = components_dict.scorers[1].params
    assert comp.executable == "path/to/executable1"  # The first one.
    # The first one, from Endpoint1, which inherited from component-level.
    assert comp.args == "--loglevel DEBUG"

    assert "externalprocess" == components_dict.scorers[2].component_type
    name, comp, transform, weights, loss_modulation = components_dict.scorers[2].params
    assert comp.properties == [
        "Property 1",  # From Endpoint1, inherited from component-level.
        "Property 2",  # From Endpoint2, overridden by endpoint.
    ]

    assert not components_dict.filters
    assert not components_dict.penalties


def test_dynamic_filter_routes_to_filters():
    """A scorer component with filter=true and a step transform is routed to filters."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [
                    {
                        "name": "MW_filter",
                        "transform": {"type": "right_step", "high": 500.0},
                    }
                ],
            }
        }
    ]

    result = get_components(components)

    assert len(result.filters) == 1
    assert result.filters[0].component_type == "molecularweight"
    assert not result.scorers
    assert not result.penalties


def test_dynamic_filter_missing_transform_raises():
    """filter=true without a transform must raise RuntimeError."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [{"name": "MW_filter"}],
            }
        }
    ]

    with pytest.raises(RuntimeError, match="must specify a transform"):
        get_components(components)


def test_dynamic_filter_non_step_transform_raises():
    """filter=true with a non-step transform (e.g. sigmoid) must raise RuntimeError."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [
                    {
                        "name": "MW_filter",
                        "transform": {"type": "sigmoid", "high": 500.0, "low": 200.0, "k": 0.5},
                    }
                ],
            }
        }
    ]

    with pytest.raises(RuntimeError, match="requires a step-type transform"):
        get_components(components)


def test_dynamic_filter_multiple_endpoints_raises():
    """filter=true with more than one endpoint must raise RuntimeError."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [
                    {
                        "name": "MW_filter_1",
                        "transform": {"type": "right_step", "high": 500.0},
                    },
                    {
                        "name": "MW_filter_2",
                        "transform": {"type": "left_step", "low": 100.0},
                    },
                ],
            }
        }
    ]

    with pytest.raises(RuntimeError, match="must have exactly one endpoint"):
        get_components(components)


def test_dynamic_filter_value_mapping_binary_routes_to_filters():
    """filter=true with a value_mapping where all values are 0 or 1 is accepted."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [
                    {
                        "name": "MW_filter",
                        "transform": {
                            "type": "value_mapping",
                            "mapping": {"small": 1, "large": 0},
                        },
                    }
                ],
            }
        }
    ]

    result = get_components(components)

    assert len(result.filters) == 1


def test_dynamic_filter_value_mapping_non_binary_raises():
    """filter=true with a value_mapping that has non-0/1 values must raise RuntimeError."""
    components = [
        {
            "MolecularWeight": {
                "filter": True,
                "endpoint": [
                    {
                        "name": "MW_filter",
                        "transform": {
                            "type": "value_mapping",
                            "mapping": {"small": 0.5, "large": 0},
                        },
                    }
                ],
            }
        }
    ]

    with pytest.raises(RuntimeError, match="do not map to 0 or 1"):
        get_components(components)


def test_collect_params():
    # Test case 1: multiple dictionaries
    params = [
        {"x": 1, "y": 1},
        {"y": 2, "z": 2},
        {"z": 3, "x": 3},
    ]
    expected_output = {
        "x": [1, None, 3],
        "y": [1, 2, None],
        "z": [None, 2, 3],
    }
    assert collect_params(params) == expected_output

    # Test case 2: Empty list
    params = []
    expected_output = {}
    assert collect_params(params) == expected_output

    # Test case 3: List with one dictionary
    params = [{"key1": "value1"}]
    expected_output = {"key1": ["value1"]}
    assert collect_params(params) == expected_output

    # Test case 4: List with dictionaries having different keys
    params = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]
    expected_output = {
        "key1": ["value1", None, None],
        "key2": [None, "value2", None],
        "key3": [None, None, "value3"],
    }
    assert collect_params(params) == expected_output
