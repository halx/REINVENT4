"""Config Validation"""

from typing import Optional
from pydantic import Field, BaseModel, ConfigDict, model_validator
from reinvent.validation import ExistingFilePath


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterSection(GlobalConfig):
    elements: list[str] = Field(default_factory=list)
    transforms: list[str] = Field(default=["standard"])
    min_heavy_atoms: int = Field(2, ge=1)
    max_heavy_atoms: int = Field(90, ge=1)
    max_mol_weight: float = Field(1200.0, ge=1.0)
    min_carbons: int = Field(2, ge=1)
    max_num_rings: int = Field(12, ge=1)
    max_ring_size: int = Field(7, ge=1)
    keep_stereo: bool = True
    keep_isotope_molecules: bool = True
    uncharge: bool = True
    canonical_tautomer: bool = False
    kekulize: bool = False
    randomize_smiles: bool = False
    report_errors: bool = False
    inchi_key_deduplicate: bool = False

    @model_validator(mode="after")
    def check_atom_count_range(self) -> "FilterSection":
        if self.min_heavy_atoms >= self.max_heavy_atoms:
            raise ValueError(
                f"min_heavy_atoms ({self.min_heavy_atoms}) must be less than "
                f"max_heavy_atoms ({self.max_heavy_atoms})"
            )
        return self


class DPLConfig(GlobalConfig):
    input_csv_file: str
    smiles_column: str = "SMILES"
    separator: str = Field("\t", min_length=1, max_length=1)  # char for polars
    output_smiles_file: str
    num_procs: Optional[int] = Field(1, ge=1)
    chunk_size: Optional[int] = Field(500, ge=1)
    filter: Optional[FilterSection] = None
    transform_file: Optional[ExistingFilePath] = None
