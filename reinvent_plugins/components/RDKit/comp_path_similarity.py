"""Tanimoto similarity using RDKit path-based (topological) fingerprints"""

from __future__ import annotations

__all__ = ["PathFPSimilarity"]

from typing import List, Literal, Optional

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass
from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint

from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent_plugins.mol_cache import molcache


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the PathFPSimilarity component.

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    smiles: List[List[str]]
    min_path: List[int]
    max_path: List[int]
    n_bits: List[int]
    aggregation: Optional[List[Literal["mean", "min", "max"]]] = Field(default=None)

    def __post_init__(self):
        if self.aggregation is None:
            self.aggregation = ["max"] * len(self.smiles)


@add_tag("__component")
class PathFPSimilarity:
    """Compute the Tanimoto similarity using RDKit path-based fingerprints.

    Scoring component to compute the Tanimoto similarity between the provided
    SMILES and the generated molecule using topological path-based fingerprints
    (RDKFingerprint).  Path-based fingerprints enumerate all linear paths up to
    ``max_path`` bonds, which can capture chain and path-based structural features
    differently from circular Morgan fingerprints.
    """

    def __init__(self, params: Parameters):
        self.fp_params = []

        for smilies, min_path, max_path, n_bits, aggregation in zip(
            params.smiles, params.min_path, params.max_path, params.n_bits, params.aggregation
        ):
            fingerprints = []

            for smi in smilies:
                mol = Chem.MolFromSmiles(smi)

                if mol is not None:
                    fingerprints.append(
                        RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=n_bits)
                    )

            if not fingerprints:
                raise ValueError(f"{__name__}: unable to convert any SMILES to fingerprints")

            self.fp_params.append((fingerprints, min_path, max_path, n_bits, aggregation))

        self.number_of_endpoints = len(params.smiles)

    _AGGREGATION_FUNCS = {
        "mean": lambda arr: np.mean(arr, axis=0),
        "min": lambda arr: np.min(arr, axis=0),
        "max": lambda arr: np.max(arr, axis=0),
    }

    @molcache
    def __call__(self, mols: List[Optional[Chem.Mol]]) -> ComponentResults:
        scores = []

        for fingerprints, min_path, max_path, n_bits, aggregation in self.fp_params:
            query_fps = [
                (
                    RDKFingerprint(mol, minPath=min_path, maxPath=max_path, fpSize=n_bits)
                    if mol is not None
                    else None
                )
                for mol in mols
            ]

            valid_mask = [fp is not None for fp in query_fps]
            valid_fps = [fp for fp in query_fps if fp is not None]
            valid_cols = [i for i, ok in enumerate(valid_mask) if ok]

            endpoint_scores = np.zeros((len(fingerprints), len(query_fps)), dtype=np.float32)

            if valid_fps:
                for i, ref_fp in enumerate(fingerprints):
                    sims = DataStructs.BulkTanimotoSimilarity(ref_fp, valid_fps)
                    endpoint_scores[i, valid_cols] = sims

            agg_func = self._AGGREGATION_FUNCS[aggregation]
            scores.append(agg_func(endpoint_scores))

        return ComponentResults(scores)
