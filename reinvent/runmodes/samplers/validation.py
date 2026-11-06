"""Config Validation"""

from typing import Literal, Optional

from reinvent.validation import GlobalConfig, ExistingFilePath, Field


class SectionParameters(GlobalConfig):
    model_file: ExistingFilePath
    num_smiles: int = Field(ge=1)
    smiles_file: Optional[ExistingFilePath] = None
    target_smiles_path: str = ""
    sample_strategy: Optional[Literal["multinomial", "beamsearch"]] = "multinomial"
    output_file: str = "samples.csv"
    target_nll_file: str = "target_nll.csv"
    unique_molecules: bool = True
    randomize_smiles: bool = True
    isomeric_smiles: bool = False
    temperature: float = Field(1.0, gt=0.0)


class SectionFilter(GlobalConfig):
    smarts: list[str]


class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class SamplingConfig(GlobalConfig):
    parameters: SectionParameters
    filter: Optional[SectionFilter] = None
    responder: Optional[SectionResponder] = None
