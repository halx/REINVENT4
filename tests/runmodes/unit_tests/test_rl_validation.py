"""Tests for run-mode validation models and TL scheduler dataclasses.

Covers:
  - SectionStage.termination Literal
  - SectionStage cross-field: min_steps < max_steps
  - SectionInception cross-field: sample_size <= memory_size
  - StepLRConfiguration.__post_init__ bounds
  - LambdaLRConfiguration.__post_init__ bounds
"""

import pytest
import pydantic

from reinvent.runmodes.RL.validation import SectionStage, SectionInception, RLConfig
from reinvent.runmodes.TL.configurations import StepLRConfiguration, LambdaLRConfiguration

# ---------------------------------------------------------------------------
# SectionStage.termination Literal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", ["simple", "null", "plateau"])
def test_section_stage_valid_termination(term):
    stage = SectionStage(max_steps=100, termination=term, scoring={})
    assert stage.termination == term


@pytest.mark.parametrize("bad", ["complex", "SIMPLE", "early_stop", ""])
def test_section_stage_invalid_termination_rejected(bad):
    with pytest.raises(pydantic.ValidationError) as exc_info:
        SectionStage(max_steps=100, termination=bad, scoring={})
    errors = exc_info.value.errors()
    assert any("termination" in str(e["loc"]) for e in errors)


# ---------------------------------------------------------------------------
# SectionStage cross-field: min_steps < max_steps
# ---------------------------------------------------------------------------


def test_section_stage_min_steps_lt_max_steps_valid():
    stage = SectionStage(max_steps=100, min_steps=10, scoring={})
    assert stage.max_steps == 100
    assert stage.min_steps == 10


def test_section_stage_min_steps_equal_max_steps_rejected():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        SectionStage(max_steps=50, min_steps=50, scoring={})
    assert "min_steps" in str(exc_info.value)


def test_section_stage_min_steps_greater_than_max_steps_rejected():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        SectionStage(max_steps=10, min_steps=50, scoring={})
    assert "min_steps" in str(exc_info.value)


def test_section_stage_max_steps_must_be_positive():
    with pytest.raises(pydantic.ValidationError):
        SectionStage(max_steps=0, scoring={})


# ---------------------------------------------------------------------------
# RLConfig cross-section: loss_modulation requires the 'dap' learning strategy
# ---------------------------------------------------------------------------


def _rl_config_dict(tmp_path, strategy_type, loss_modulation):
    prior = tmp_path / "prior.ckpt"
    prior.write_text("x")
    agent = tmp_path / "agent.ckpt"
    agent.write_text("x")
    return {
        "parameters": {"prior_file": str(prior), "agent_file": str(agent)},
        "learning_strategy": {"type": strategy_type},
        "stage": [
            {
                "max_steps": 100,
                "scoring": {
                    "component": [
                        {
                            "MaxReferenceSimilarity": {
                                "endpoint": [
                                    {"name": "x", "weight": 1.0, "loss_modulation": loss_modulation}
                                ]
                            }
                        }
                    ]
                },
            }
        ],
    }


def test_loss_modulation_with_non_dap_strategy_rejected(tmp_path):
    with pytest.raises(pydantic.ValidationError, match="dap"):
        RLConfig(**_rl_config_dict(tmp_path, "mascof", True))


def test_loss_modulation_with_dap_strategy_allowed(tmp_path):
    RLConfig(**_rl_config_dict(tmp_path, "dap", True))  # must not raise


def test_non_dap_without_loss_modulation_allowed(tmp_path):
    RLConfig(**_rl_config_dict(tmp_path, "mascof", False))  # must not raise


# ---------------------------------------------------------------------------
# SectionInception cross-field: sample_size <= memory_size
# ---------------------------------------------------------------------------


def test_section_inception_valid_defaults():
    inc = SectionInception()
    assert inc.sample_size <= inc.memory_size


def test_section_inception_sample_size_le_memory_size_valid():
    inc = SectionInception(memory_size=50, sample_size=10)
    assert inc.sample_size == 10


def test_section_inception_sample_size_equal_memory_size_valid():
    inc = SectionInception(memory_size=20, sample_size=20)
    assert inc.sample_size == inc.memory_size


def test_section_inception_sample_size_gt_memory_size_rejected():
    with pytest.raises(pydantic.ValidationError) as exc_info:
        SectionInception(memory_size=10, sample_size=20)
    assert "sample_size" in str(exc_info.value)


def test_section_inception_memory_size_ge_1():
    with pytest.raises(pydantic.ValidationError):
        SectionInception(memory_size=0)


def test_section_inception_sample_size_ge_1():
    with pytest.raises(pydantic.ValidationError):
        SectionInception(sample_size=0)


# ---------------------------------------------------------------------------
# StepLRConfiguration bounds
# ---------------------------------------------------------------------------


def test_step_lr_valid_defaults():
    cfg = StepLRConfiguration()
    assert cfg.lr > 0
    assert 0 < cfg.gamma <= 1


@pytest.mark.parametrize("lr", [0.0, -0.001, -1.0])
def test_step_lr_nonpositive_lr_rejected(lr):
    with pytest.raises(ValueError, match="lr"):
        StepLRConfiguration(lr=lr)


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.1, 2.0])
def test_step_lr_invalid_gamma_rejected(gamma):
    with pytest.raises(ValueError, match="gamma"):
        StepLRConfiguration(gamma=gamma)


def test_step_lr_gamma_1_accepted():
    cfg = StepLRConfiguration(gamma=1.0)
    assert cfg.gamma == 1.0


def test_step_lr_step_ge_1():
    with pytest.raises(ValueError, match="step"):
        StepLRConfiguration(step=0)


# ---------------------------------------------------------------------------
# LambdaLRConfiguration bounds
# ---------------------------------------------------------------------------


def test_lambda_lr_valid_defaults():
    cfg = LambdaLRConfiguration()
    assert cfg.lr > 0
    assert cfg.warmup > 0


def test_lambda_lr_warmup_zero_rejected():
    with pytest.raises(ValueError, match="warmup"):
        LambdaLRConfiguration(warmup=0)


def test_lambda_lr_negative_warmup_rejected():
    with pytest.raises(ValueError, match="warmup"):
        LambdaLRConfiguration(warmup=-100)


@pytest.mark.parametrize("lr", [0.0, -1e-5])
def test_lambda_lr_nonpositive_lr_rejected(lr):
    with pytest.raises(ValueError, match="lr"):
        LambdaLRConfiguration(lr=lr)


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.0, 1.5])
def test_lambda_lr_invalid_beta1_rejected(beta):
    with pytest.raises(ValueError, match="beta1"):
        LambdaLRConfiguration(beta1=beta)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_lambda_lr_invalid_beta2_rejected(beta):
    with pytest.raises(ValueError, match="beta2"):
        LambdaLRConfiguration(beta2=beta)


def test_lambda_lr_eps_zero_rejected():
    with pytest.raises(ValueError, match="eps"):
        LambdaLRConfiguration(eps=0.0)
