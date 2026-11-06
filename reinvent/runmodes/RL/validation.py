"""Config Validation"""

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from reinvent.validation import GlobalConfig, ExistingFilePath, WritableFilePath


class SectionParameters(GlobalConfig):
    prior_file: ExistingFilePath
    agent_file: ExistingFilePath
    summary_csv_prefix: str = "summary"
    use_checkpoint: bool = False
    purge_memories: bool = True
    smiles_file: Optional[ExistingFilePath] = None  # not Reinvent
    sample_strategy: Optional[Literal["multinomial", "beamsearch"]] = "multinomial"
    distance_threshold: int = 99999  # Transformer
    batch_size: int = Field(100, ge=1)
    randomize_smiles: bool = True
    isomeric_smiles: bool = False
    temperature: float = Field(1.0, gt=0.0)
    tb_isim: Optional[bool] = False  # Add iSIM tracking as optional parameter


class SectionLearningStrategy(GlobalConfig):
    type: Literal[
        "dap",
        "sdap",
        "mauli",
        "mascof",
        "dap_reinforce",
        "mascof_reinforce",
        "mauli_reinforce",
    ] = "dap"
    sigma: int = Field(128, ge=1)
    rate: float = Field(0.0001, gt=0.0)


class SectionDiversityFilter(GlobalConfig):
    type: Literal[
        "IdenticalMurckoScaffold",
        "IdenticalTopologicalScaffold",
        "ScaffoldSimilarity",
        "PenalizeSameSmiles",
        "BitBirchDiversityFilter",
    ]
    bucket_size: int = Field(25, ge=1)  # not needed for PenalizeSameSmiles
    minscore: float = Field(0.4, ge=0.0, le=1.0)  # not needed for PenalizeSameSmiles
    minsimilarity: Optional[float] = Field(0.4, ge=0.0, le=1.0)  # ScaffoldSimilarity only
    penalty_multiplier: Optional[float] = Field(0.5, ge=0.0, le=1.0)  # PenalizeSameSmiles only
    penalty_function: Literal["Step", "Linear", "Tanh", "Sigmoid", "Erf"] = Field("Step")
    intrinsic_reward: Optional[Literal["RND", "Information"]] = Field(None)
    # BitBirchDiversityFilter parameters
    discard: bool = Field(False)
    merge_threshold: float = Field(0.65, ge=0.0, le=1.0)
    branching_factor: int = Field(2500, ge=1)
    recluster_tolerance: Optional[float] = Field(None, ge=0.0)
    recluster_interval: Optional[int] = Field(None, ge=1)
    debug: bool = False
    learning_rate: float = Field(1e-4, gt=0.0)


class SectionIntrinsicPenalty(GlobalConfig):
    type: Literal["IdenticalMurckoScaffoldRND", "InformationPenalty"]
    penalty_function: Literal["Linear", "Tanh", "Step", "Sigmoid", "Erf"]
    bucket_size: int = Field(25, ge=1)
    minscore: float = Field(0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(
        1e-4,
        gt=0.0,
    )  # IdenticalScaffoldPenaltyRND only, defaults to 1e-4


class SectionInception(GlobalConfig):
    smiles_file: Optional[ExistingFilePath] = None
    memory_size: int = Field(50, ge=1)
    sample_size: int = Field(10, ge=1)
    deduplicate: bool = True  # obsolete, True by default now
    diversity_penalty_weight: float = Field(0.0, ge=0.0, le=1.0)
    is_weighted_sampling: bool = False
    is_weighted_scores: bool = False
    is_weight_clip: Optional[float] = None
    is_weight_temperature: float = 1.0
    debug: bool = False
    relevance_threshold: float = Field(0.6, ge=0.0, le=1.0)  # loss-modulation admission gate

    @model_validator(mode="after")
    def sample_size_le_memory_size(self) -> "SectionInception":
        if self.sample_size > self.memory_size:
            raise ValueError(
                f"sample_size ({self.sample_size}) must be <= memory_size ({self.memory_size})"
            )
        return self


class SectionStage(GlobalConfig):
    max_steps: int = Field(ge=1)
    max_score: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    chkpt_file: Optional[WritableFilePath] = None
    termination: Literal["simple", "null", "plateau"] = "simple"
    min_steps: Optional[int] = Field(50, ge=0)
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
    diversity_filter: Optional[SectionDiversityFilter] = None
    loss_modulation_aggregation: Optional[str] = None  # stage override of the global default

    @model_validator(mode="after")
    def min_steps_lt_max_steps(self) -> "SectionStage":
        if self.min_steps is not None and self.min_steps >= self.max_steps:
            raise ValueError(
                f"min_steps ({self.min_steps}) must be less than max_steps ({self.max_steps})"
            )
        return self


# FIXME: may only need this once
class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class RLConfig(GlobalConfig):
    parameters: SectionParameters
    stage: List[SectionStage]
    learning_strategy: Optional[SectionLearningStrategy] = None
    diversity_filter: Optional[SectionDiversityFilter] = None
    intrinsic_penalty: Optional[SectionIntrinsicPenalty] = None
    inception: Optional[SectionInception] = None
    responder: Optional[SectionResponder] = None
    loss_modulation_aggregation: str = "arithmetic"  # global default; stages may override

    @model_validator(mode="after")
    def loss_modulation_requires_dap(self) -> "RLConfig":
        """Loss modulation needs a per-molecule loss, i.e. the 'dap' learning strategy.

        The scalar-loss strategies (mascof/mauli/sdap and the reinforce variants) reduce the
        loss before it can be modulated per molecule, so reject the combination at config read.
        """
        strategy_type = self.learning_strategy.type if self.learning_strategy else "dap"
        if strategy_type == "dap":
            return self

        for stage in self.stage:
            for component in stage.scoring.get("component", []):
                if not component:
                    continue
                _, component_value = next(iter(component.items()))
                for endpoint in component_value.get("endpoint", []):
                    if endpoint.get("loss_modulation", False):
                        raise ValueError(
                            f"loss_modulation is enabled on endpoint "
                            f"'{endpoint.get('name', '<unnamed>')}' but learning_strategy.type is "
                            f"'{strategy_type}'; per-molecule loss modulation requires "
                            f"learning_strategy.type = 'dap'"
                        )
        return self
