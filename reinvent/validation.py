"""Global Config Validation"""

import re
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Reusable annotated path types
# ---------------------------------------------------------------------------


def _check_file_exists(v: str) -> str:
    if not Path(v).is_file():
        raise ValueError(f"File not found: '{v}'")
    return v


def _check_parent_dir_exists(v: str) -> str:
    if not Path(v).parent.is_dir():
        raise ValueError(f"Parent directory does not exist: '{Path(v).parent}'")
    return v


ExistingFilePath = Annotated[str, AfterValidator(_check_file_exists)]
WritableFilePath = Annotated[str, AfterValidator(_check_parent_dir_exists)]

from .version import __config_version__ as _CONFIG_VERSION

# cpu, cuda[:N], mps[:N], xpu[:N] (Intel ARC/oneAPI)
_DEVICE_RE = re.compile(r"^(cpu|cuda|mps|xpu)(:\d+)?$")


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ReinventConfig(GlobalConfig):
    run_type: Literal[
        "sampling",
        "staged_learning",
        "transfer_learning",
        "scoring",
        "enumeration",
    ]
    device: str = "cpu"
    version: int = Field(default=_CONFIG_VERSION, ge=1)
    tb_logdir: Optional[str] = None
    json_out_config: Optional[str] = None
    seed: Optional[int] = None
    parameters: dict

    # run mode dependent
    scoring: Optional[dict] = None  # RL, scoring
    scheduler: Optional[dict] = None  # TL
    responder: Optional[dict] = None  # Rl, TL, sampling
    filter: Optional[dict] = None  # sampling

    # RL
    stage: Optional[list] = None
    learning_strategy: Optional[dict] = None
    diversity_filter: Optional[dict] = None
    intrinsic_penalty: Optional[dict] = None
    inception: Optional[dict] = None

    @field_validator("device")
    @classmethod
    def device_must_be_valid(cls, v: str) -> str:
        if not _DEVICE_RE.match(v):
            raise ValueError(
                f"Invalid device '{v}'. Must be 'cpu', 'cuda', 'mps', 'xpu' "
                f"(Intel ARC), or an indexed variant such as 'cuda:0'."
            )
        return v
