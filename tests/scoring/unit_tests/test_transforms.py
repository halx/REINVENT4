import pytest

import numpy as np

from reinvent.scoring.transforms import DoubleSigmoid
from reinvent.scoring.transforms import ReverseSigmoid
from reinvent.scoring.transforms import Sigmoid
from reinvent.scoring.transforms import LeftStep
from reinvent.scoring.transforms import RightStep
from reinvent.scoring.transforms import Step
from reinvent.scoring.transforms import ValueMapping
from reinvent.scoring.transforms import ExponentialDecay
from reinvent.scoring.transforms import SymmetricExponentialDecay


@pytest.mark.parametrize(
    "low, high, coef_div, coef_si, coef_se",
    [(-5.0, 5.0, 100.0, 150.0, 150.0), (4.0, 9.0, 35.0, 150.0, 150.0)],
)
def test_double_sigmoid(low, high, coef_div, coef_si, coef_se):
    from reinvent.scoring.transforms.double_sigmoid import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="", low=low, high=high, coef_div=coef_div, coef_si=coef_si, coef_se=coef_se
    )
    transform = DoubleSigmoid(params)
    results = transform(data)
    max_idx = np.argmax(results)
    assert np.all(results[1:max_idx] >= results[0 : max_idx - 1])
    assert np.all(results[max_idx:-1] >= results[max_idx + 1 :])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize(
    "low, high, k",
    [
        (-5.0, 5.0, 1.0),
        (0.0, 10.0, 0.5),  # positive range, gentler slope
        (-10.0, 0.0, 2.0),  # negative range, steeper slope
    ],
)
def test_reverse_sigmoid(low, high, k):
    from reinvent.scoring.transforms.sigmoids import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="",
        low=low,
        high=high,
        k=k,
    )
    transform = ReverseSigmoid(params)
    results = transform(data)
    assert np.all(results[1:] <= results[:-1])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize(
    "low, high, k",
    [
        (-5.0, 5.0, 1.0),
        (0.0, 10.0, 0.5),  # positive range, gentler slope
        (-10.0, 0.0, 2.0),  # negative range, steeper slope
    ],
)
def test_sigmoid(low, high, k):
    from reinvent.scoring.transforms.sigmoids import Parameters

    data = np.linspace(-20, 20, 21, dtype=np.float32)
    params = Parameters(
        type="",
        low=low,
        high=high,
        k=k,
    )
    transform = Sigmoid(params)
    results = transform(data)
    assert np.all(results[1:] >= results[:-1])
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)


@pytest.mark.parametrize("mapping", [{0: 1, 1: 2, 2: 5}])
def test_value_mapping(mapping):
    from reinvent.scoring.transforms.value_mapping import Parameters

    data = [0, 0, 1, 2, 1, 2]
    params = Parameters(type="", mapping=mapping)
    transform = ValueMapping(params)
    results = transform(data)
    assert np.all(results == [1, 1, 2, 5, 2, 5])


@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 3),
        (0, 0),  # degenerate: single-point boundary
        (10, 20),  # both positive
    ],
)
def test_left_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = LeftStep(params)
    results = transform(data)
    assert np.all(results == (data <= low))


@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 3),
        (0, 0),  # degenerate: single-point boundary
        (-20, 10),  # wide range
    ],
)
def test_right_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = RightStep(params)
    results = transform(data)
    assert np.all(results == (data >= high))


@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 3),
        (-20, 20),  # all data in range
        (5, 5),  # degenerate: only exact value passes
    ],
)
def test_step(low, high):
    from reinvent.scoring.transforms.steps import Parameters

    data = np.linspace(-20, 20, 100)
    params = Parameters(
        type="",
        low=low,
        high=high,
    )
    transform = Step(params)
    results = transform(data)
    assert np.all(results == ((data <= high) & (data >= low)))


@pytest.mark.parametrize("k", [1.0, 10])
def test_exponential_decay(k):
    from reinvent.scoring.transforms.exponential_decay import Parameters

    data = np.linspace(-5, 5, 11, dtype=np.float32)
    params = Parameters(
        type="",
        k=k,
    )
    transform = ExponentialDecay(params)
    results = transform(data)
    assert np.all(np.diff(results) <= 0)  # Check that the function is monotonically decreasing.
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)
    assert results[0] == 1.0
    assert results[-1] <= 0.01


@pytest.mark.parametrize("target, k", [(0.0, 1.0), (2.5, 0.5)])
def test_symmetric_exponential_decay(target, k):
    from reinvent.scoring.transforms.symmetric_exponential_decay import Parameters

    data = np.append(np.linspace(-5, 5, 11, dtype=np.float32), target)
    params = Parameters(type="", target=target, k=k)
    transform = SymmetricExponentialDecay(params)
    results = transform(data)

    # Peak is exactly 1.0 at target
    exact_idx = np.argmin(np.abs(data - target))
    assert results[exact_idx] == pytest.approx(1.0)

    # Bounded in [0, 1]
    assert np.all(results <= 1.0)
    assert np.all(results >= 0.0)

    # Monotonically non-increasing as distance from target grows
    deltas = np.abs(data - target)
    order = np.argsort(deltas)
    assert np.all(np.diff(results[order]) <= 1e-6)

    # Symmetric: points equidistant from target score the same
    left = np.float32(target - 1.0)
    right = np.float32(target + 1.0)
    left_score = transform(np.array([left]))[0]
    right_score = transform(np.array([right]))[0]
    assert left_score == pytest.approx(right_score)


def test_symmetric_exponential_decay_invalid_k():
    from reinvent.scoring.transforms.symmetric_exponential_decay import Parameters

    params = Parameters(type="", target=0.0, k=-1.0)
    with pytest.raises(ValueError, match="k must be > 0"):
        SymmetricExponentialDecay(params)
