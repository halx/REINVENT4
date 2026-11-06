"""Dataclasses to hold transformed and aggregated result from components."""

from __future__ import annotations

__all__ = ["ComponentResults", "SmilesAssociatedComponentResults"]
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class SmilesResult:
    """Container for scores, uncertainty, relevance, failure data and metadata for a single smiles

    relevance is an optional per-endpoint value in [0,1] (1 = fully relevant, 0 = ignore) used by
    loss modulation; it is None for SMILES a component did not produce a relevance for.
    """

    score: tuple  # multiple scores per component, and can be numbers or strings (as in MMP)
    score_property: Optional[List[Dict]] = None
    uncertainities: Optional[float] = None
    metadata: Optional[Dict] = None
    failures_properties: Optional[Dict] = None
    relevance: Optional[tuple] = (
        None  # per-endpoint relevance in [0,1], 1=fully relevant (loss modulation)
    )


@dataclass
class ComponentResults:
    """Container for the scores, uncertainties and metadata

    At the minimum the scores must be provided.  The order of the score array
    must be the same as the order of SMILES passed to each component.  Failure
    of computation of score must be indicated by NaN. Do not use zero for this!
    scores_properties can be used to pass on metadata on the scores
    uncertainty_type is currently assumed to be the same for all values
    failure_properties can be used to provide details on the failure of a component
    meta_data is a general facility to pass on metadata
    relevance is an optional per-endpoint value in [0,1] (1 = fully relevant, 0 = ignore), one array
    per endpoint aligned with scores, used by the loss modulation strategy
    """

    scores: List[np.ndarray]
    scores_properties: Optional[List[Dict]] = None
    uncertainty: Optional[List[np.ndarray]] = None
    uncertainty_type: Optional[str] = None
    uncertainty_properties: Optional[List[Dict]] = None
    failures_properties: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None
    # optional per-endpoint relevance in [0,1] (1 = fully relevant, 0 = ignore) for loss modulation
    relevance: Optional[List[np.ndarray]] = None


@dataclass
class SmilesAssociatedComponentResults:
    """Container for the scores, uncertainties and metadata, indexed by smiles.

    This class is built on top of ComponentResults to create an object with smiles-based indexing
    At the minimum the scores must be provided.
    Failure of computation of score must be indicated by NaN. Do not use zero for this!
    uncertainty_type is currently assumed to be the same for all values
    failure_properties can be used to provide details on the failure of a component
    meta_data is a general facility to pass on metadata per smiles
    relevance, when present, holds per-endpoint value in [0,1] (1 = fully relevant, 0 = ignore) for loss modulation


    """

    # this is the main holder for smiles-associated results
    data: Dict[str, SmilesResult] = field(default=None)

    # this the base ComponentResults input
    component_results: Optional[ComponentResults] = field(init=False, default=None)

    # uncertainty type could be set a component level
    uncertainty_type: Optional[str] = field(default=None)

    # metadata keys declared by the scoring component at init time
    _declared_metadata_keys: Tuple[str, ...] = field(init=False, default=())

    @staticmethod
    def _score_lists_to_dict(
        smiles: List[str],
        scores: List[List[float | str]],
        metadata: Optional[Dict[str, List[float | str]]] = None,
        relevance: Optional[List[List[float]]] = None,
    ) -> dict[str, SmilesResult]:
        """Utility method for converting from scores in the form of [[score0_mol0,score0_mol1,...],
                                                                      [score1_mol0,score1_mol2,...]]]

        and possibly metadata of the form Dict[metadata0_name:[metadata0_mol0,metadata0_mol1,metadata0_mol2,...]]
        as returned by most scoring functions to dictionaries {smiles:SmilesResult((score_1,score2)}

        relevance, when provided, uses the same [[endpoint0_mol0,...],[endpoint1_mol0,...]] layout as
        scores and is stored per SMILES for loss modulation; SMILES without one get None.
        """
        values_to_upack = [smiles, list(zip(*scores))]

        if metadata:
            # if we have metadata, we need to arrange it by smiles for each key.

            values_to_upack.append(
                [
                    dict(zip(list(metadata.keys()), value_group))
                    for value_group in list(zip(*metadata.values()))
                ]
            )
        else:
            # Metadata is optional so we need an empty placeholder
            values_to_upack.append([dict()] * len(smiles))

        # relevance, when present, has one list per endpoint (same layout as scores)
        if relevance is not None:
            values_to_upack.append(list(zip(*relevance)))
        else:
            values_to_upack.append([None] * len(smiles))

        return {
            smiles: SmilesResult(score=score, metadata=metadata, relevance=relevance)
            for smiles, score, metadata, relevance in zip(*values_to_upack)
        }

    def get_metadata_names(self):
        if self._declared_metadata_keys:
            return self._declared_metadata_keys
        # fallback for legacy components without declared metadata_keys
        metadata_names = set()
        for smile in self.data.keys():
            metadata_dictionary = self.data[smile].metadata
            for key in metadata_dictionary.keys():
                metadata_names.add(key)
        return metadata_names

    def __init__(
        self, component_results: ComponentResults = None, smiles: list[srr] = None, data=None
    ):
        """Constructor from either ComponentResults and smiles,  or setting th data directly

        When built from a ComponentResults, its optional per-endpoint relevance is carried through.
        """
        self._declared_metadata_keys = ()
        if data is None:
            self.data = SmilesAssociatedComponentResults._score_lists_to_dict(
                smiles,
                component_results.scores,
                component_results.metadata,
                relevance=component_results.relevance,
            )
        else:
            self.data = data

    @classmethod
    def create_from_scores(
        cls, smiles: list[str], scores: List[List]
    ) -> SmilesAssociatedComponentResults:
        """Method to create ComponentResults from a list of list of scores, as in components with multiple endpoints"""
        return cls([], data=SmilesAssociatedComponentResults._score_lists_to_dict(smiles, scores))

    def update_scores(self, smiles: List[str], scores: List[List[float | str]]) -> None:
        """Method to update ComponentResults object from a list of list of scores, as in components with multiple endpoints"""
        self.data.update(SmilesAssociatedComponentResults._score_lists_to_dict(smiles, scores))

    def fetch_scores(self, smiles: List[str], transpose=False) -> List:
        """Method to retrive a list of scores for a given list of smiles

        :param smiles: list of SMILES
        :param transpose: bool, if False dimensions are SMILES x endpoints, if True dimensions are endpoints x SMILES
        :returns: list of list of scores

        """
        # note that the input smiles are needed to ensure the list has a consistent order with validity masked in scorer
        # any missing SMILES get NaN scores so reporters do not crash on incomplete results
        if self.data:
            n_endpoints = len(next(iter(self.data.values())).score)
        else:
            n_endpoints = 1

        scores_list = []
        for smile in smiles:
            if smile in self.data:
                scores_list.append(self.data[smile].score)
            else:
                scores_list.append((np.nan,) * n_endpoints)

        if transpose:  # original csv writing and reporting functions need the data in this format
            scores_list = list(zip(*scores_list))  # first dimension is endpoints

        return scores_list

    def fetch_metadata(self, smiles: List[str], transpose=False) -> dict:
        """Method to retrieve a dictionary of available metadata for a given list of smiles

        :param smiles: list of SMILES
        :param transpose: bool, if False dimensions are SMILES x endpoints, if True dimensions are endpoints x SMILES
        :returns: dictionary with metadata. Keys are metadata names, values are list of length SMILES

        """
        # first, update all metadata names available
        metadata_names = self.get_metadata_names()
        metadata_collection = dict()

        for metadata_name in metadata_names:
            metadata_collection[metadata_name] = []

            for smile in smiles:
                smile_metadata = self[smile].metadata if smile in self.data else None

                if smile_metadata and metadata_name in smile_metadata.keys():
                    metadata_collection[metadata_name].append(str(smile_metadata[metadata_name]))
                else:
                    metadata_collection[metadata_name].append(None)

        return metadata_collection

    def fetch_relevance(self, smiles: List[str], transpose=False) -> Optional[List]:
        """Retrieve per-endpoint relevance for a list of SMILES, or None if none is available.

        SMILES without a relevance (e.g. masked/invalid) are filled with NaN to keep shapes aligned.
        """
        rel_list = [self[s].relevance if self[s] is not None else None for s in smiles]

        if all(c is None for c in rel_list):
            return None

        n_endpoints = len(next(c for c in rel_list if c is not None))
        nan_fill = (float("nan"),) * n_endpoints
        filled = [c if c is not None else nan_fill for c in rel_list]

        if transpose:
            return list(zip(*filled))

        return filled

    def __getitem__(self, smiles: str) -> SmilesResult:
        """Retrieve all results data for a single smile strings"""

        return self.data[smiles] if smiles in self.data else None
