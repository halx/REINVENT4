"""Unit tests for the five BasePenalty subclasses introduced in PR #353."""

import pytest

from reinvent.runmodes.RL.memories.penalties.step_penalty import StepPenalty
from reinvent.runmodes.RL.memories.penalties.linear_penalty import LinearPenalty
from reinvent.runmodes.RL.memories.penalties.tanh_penalty import TanhPenalty
from reinvent.runmodes.RL.memories.penalties.sigmoid_penalty import SigmoidPenalty
from reinvent.runmodes.RL.memories.penalties.erf_penalty import ErfPenalty

BUCKET_SIZE = 10


# ---------------------------------------------------------------------------
# StepPenalty
# ---------------------------------------------------------------------------


class TestStepPenalty:
    def setup_method(self):
        self.p = StepPenalty(BUCKET_SIZE)

    def test_zero_utilization_returns_one(self):
        assert self.p(0) == 1.0

    def test_at_bucket_size_still_allowed(self):
        # The bucket is full at bucket_size; the molecule that fills it should
        # NOT be zeroed — only the one that exceeds it.  This validates the > fix.
        assert self.p(BUCKET_SIZE) == 1.0

    def test_one_over_bucket_size_zeroed(self):
        assert self.p(BUCKET_SIZE + 1) == 0.0

    def test_well_above_bucket_size_zeroed(self):
        assert self.p(BUCKET_SIZE * 3) == 0.0

    def test_return_is_in_unit_interval(self):
        for u in range(0, BUCKET_SIZE * 2):
            assert 0.0 <= self.p(u) <= 1.0


# ---------------------------------------------------------------------------
# LinearPenalty
# ---------------------------------------------------------------------------


class TestLinearPenalty:
    def setup_method(self):
        self.p = LinearPenalty(BUCKET_SIZE)

    def test_zero_utilization_returns_one(self):
        assert self.p(0) == pytest.approx(1.0)

    def test_at_bucket_size_is_zero(self):
        assert self.p(BUCKET_SIZE) == pytest.approx(0.0)

    def test_halfway_is_half(self):
        assert self.p(BUCKET_SIZE // 2) == pytest.approx(0.5)

    def test_above_bucket_size_clamped_to_zero(self):
        assert self.p(BUCKET_SIZE + 5) == pytest.approx(0.0)

    def test_return_is_in_unit_interval(self):
        for u in range(0, BUCKET_SIZE * 2):
            assert 0.0 <= self.p(u) <= 1.0


# ---------------------------------------------------------------------------
# TanhPenalty
# ---------------------------------------------------------------------------


class TestTanhPenalty:
    def setup_method(self):
        self.p = TanhPenalty(BUCKET_SIZE)

    def test_zero_utilization_close_to_one(self):
        assert self.p(0) > 0.9

    def test_at_bucket_size_small(self):
        assert self.p(BUCKET_SIZE) < 0.2

    def test_monotonically_decreasing(self):
        values = [self.p(u) for u in range(BUCKET_SIZE + 1)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_return_is_non_negative(self):
        for u in range(0, BUCKET_SIZE * 2):
            assert self.p(u) >= 0.0


# ---------------------------------------------------------------------------
# SigmoidPenalty
# ---------------------------------------------------------------------------


class TestSigmoidPenalty:
    def setup_method(self):
        self.p = SigmoidPenalty(BUCKET_SIZE)

    def test_zero_utilization_close_to_one(self):
        assert self.p(0) > 0.9

    def test_at_bucket_size_small(self):
        assert self.p(BUCKET_SIZE) < 0.2

    def test_monotonically_decreasing(self):
        values = [self.p(u) for u in range(BUCKET_SIZE + 1)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_return_is_in_unit_interval(self):
        for u in range(0, BUCKET_SIZE * 2):
            assert 0.0 <= self.p(u) <= 1.0


# ---------------------------------------------------------------------------
# ErfPenalty
# ---------------------------------------------------------------------------


class TestErfPenalty:
    def setup_method(self):
        self.p = ErfPenalty(BUCKET_SIZE)

    def test_zero_utilization_close_to_one(self):
        assert self.p(0) > 0.9

    def test_at_bucket_size_small(self):
        assert self.p(BUCKET_SIZE) < 0.25

    def test_monotonically_decreasing(self):
        values = [self.p(u) for u in range(BUCKET_SIZE + 1)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_return_is_non_negative(self):
        for u in range(0, BUCKET_SIZE * 2):
            assert self.p(u) >= 0.0
