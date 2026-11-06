# from __future__ import annotations
"""The Pepinvent enumeration module"""

__all__ = ["AAEnumerator"]

import csv
import itertools
import os
from typing import List, Tuple, Union, Dict
import logging

import pandas as pd
import torch
from rdkit import Chem
from reinvent.runmodes.enumeration.validation import AAEnumerationConfig

from reinvent.runmodes.samplers.sampler import Sampler, validate_smiles
from reinvent.runmodes.samplers import params
from reinvent.models.model_factory.sample_batch import SampleBatch, BatchRow
from reinvent.chemistry.tokens import PEPINVENT_CHUCKLES_SEPARATOR_TOKEN, PEPINVENT_MASK_TOKEN
from reinvent.runmodes.scoring.file_io import TabFileReader
from reinvent.runmodes.scoring.run_scoring import REINVENT_SMILES_COLUMN
from reinvent.utils.config_parse import smiles_func

logger = logging.getLogger(__name__)


class AAEnumerator:
    """Carry out enumeration for Peptides"""

    def __init__(self, input_config: AAEnumerationConfig):
        self.config = input_config

        aa_library_filename = os.path.abspath(input_config.parameters.amino_acid_library_file)
        self.library = self._load_library(aa_library_filename)
        self.fillers = list(self.library.keys())  # get amino acid names from the library

        peptide_filename = os.path.abspath(input_config.parameters.smiles_file)
        # reader = read_data(peptide_filename, smiles_column='0', actions=None)
        self.input_smilies = read_smiles_csv_file(peptide_filename, 0)
        logger.info(f"Input peptide read from file {input_config.parameters.smiles_file}")

        # self.iterator = self._create_iterator(reader.smilies)
        self.iterator = self._create_iterator(self.input_smilies)

    def _create_iterator(self, smilies: List[str]) -> itertools.product:
        """Create an iterator for the enumeration"""
        input_masked_peptide = smilies[0]
        self.num_mask = input_masked_peptide.count(PEPINVENT_MASK_TOKEN)

        if self.num_mask == 0:
            raise ValueError("No masked amino acids found in the input peptide.")
        elif self.num_mask <= 2:
            iterator = itertools.product(self.fillers, repeat=self.num_mask)
        else:
            raise ValueError("Enumeration is limited to 2 or less masked amino acids.")
        return iterator

    def _load_library(self, file_path: str) -> Dict:
        """Load the amino acid library from a file"""
        smiles_column = self.config.parameters.smiles_column
        names_column = self.config.parameters.aa_names_column
        if smiles_column == REINVENT_SMILES_COLUMN:
            raise RuntimeError(f"{__name__}: the column name {REINVENT_SMILES_COLUMN} is reserved")

        logger.info(f"Loading amino acid library from {file_path}")
        library_data = pd.read_csv(file_path)
        library_data = library_data[[names_column, smiles_column]]
        library_data = library_data.set_index(names_column).to_dict()[smiles_column]
        library_data = {k.strip(): v.strip() for k, v in library_data.items()}  # clean whitespace
        logger.info("Amino acid library loaded successfully")
        return library_data

    def sample(self, batch_size: int) -> SampleBatch:
        """Samples the Pepinvent model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
        """
        # Iterator doesnt need [start, stop] updates
        start = 0
        stop = batch_size
        logger.info(f"Sampling {batch_size} peptides")

        aa_smi_sampled = list(
            map(self._convert_to_smiles_output, itertools.islice(self.iterator, start, stop))
        )
        sample_batch_input = self.input_smilies * batch_size
        fake_nlls = torch.zeros(batch_size)

        sampled = SampleBatch(items1=sample_batch_input, items2=aa_smi_sampled, nlls=fake_nlls)

        mols, sampled = self._join_fragments(sampled)

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.smilies, isomeric=True, return_original_smiles=True
        )

        return sampled

    def _convert_to_smiles_output(self, aa_ids: Union[str, Tuple[str]]) -> str:
        """Convert a list of amino acid names to PepINVENT-like output (in Chuckles format)"""
        if isinstance(aa_ids, str):
            aa_smiles = self.library.get(aa_ids, "")
            aa_output = aa_smiles[:-1]
        else:
            aa_smiles = [self.library.get(idx, "") for idx in aa_ids]
            aa_smiles = [aa[:-1] for aa in aa_smiles]
            aa_output = "|".join(aa_smiles)

        return aa_output

    def _join_fragments(self, sequences: SampleBatch) -> Tuple[List[Chem.rdchem.Mol], SampleBatch]:
        """Join input masked peptide with generated fillers

        :param sequences: a batch of sequences
        :returns: a list of RDKit molecules and SampleBatch where smilies field is joined complete smiles
        """

        mols = []
        samples = []
        for sample in sequences:
            smiles = sample.input
            num_fillers = sample.output.count("|") + 1
            num_mask = smiles.count("?")
            # The number of fillers generated is less than the number of masked slots
            if num_fillers < num_mask:
                mol = None
            # The number of fillers generated is greater than the number of masked slots
            elif num_fillers > num_mask:
                # Ignore the extra generated amino acids
                sample.output = "|".join(
                    sample.output.split(PEPINVENT_CHUCKLES_SEPARATOR_TOKEN)[:num_mask]
                )
                mol, complete_smiles = self._create_complete_mol(sample)
                sample.smiles = complete_smiles
            # The number of fillers generated is equal to the number of masked slots
            else:  # elif num_fillers == num_mask:
                mol, complete_smiles = self._create_complete_mol(sample)
                sample.smiles = complete_smiles

            mols.append(mol)
            samples.append(sample)

        sampled = SampleBatch.from_list(samples)

        return mols, sampled

    def _create_complete_mol(self, sample: BatchRow) -> Tuple[Chem.rdchem.Mol, str]:
        smiles = sample.input
        # Put filler in the masked position
        for replacement in sample.output.split(PEPINVENT_CHUCKLES_SEPARATOR_TOKEN):
            smiles = smiles.replace(PEPINVENT_MASK_TOKEN, replacement, 1)
        # replace the chuckles separator token with an empty string
        complete_smiles = smiles.replace(PEPINVENT_CHUCKLES_SEPARATOR_TOKEN, "")

        return Chem.rdmolfiles.MolFromSmiles(complete_smiles), complete_smiles


def read_smiles_csv_file(
    filename: str,
    columns: Union[int, slice],
    delimiter: str = "\t",
    actions: List[smiles_func] = None,
    remove_duplicates: bool = False,
) -> Union[List[str], List[Tuple]]:
    """Read a SMILES column from a CSV file

    :param filename: name of the CSV file
    :param columns: what number of the column to extract (TL reads 2, RL/sampling 1)
    :param delimiter: column delimiter, must be a single character
    :param actions: a list of callables that act on each SMILES (only Reinvent
                    and Mol2Mol)
    :param remove_duplicates: whether to remove duplicates
    :returns: a list of SMILES or a list of a tuple of SMILES
    """

    lower = filename.lower()
    if lower.endswith(".csv"):
        header = True
    elif lower.endswith(".smi"):
        header = False
    else:
        raise RuntimeError(f"Unknown file format: {filename} (expected .csv or .smi)")
    # reader = TabFileReader(filename, header=header, actions=actions, smiles_column='')
    # reader.read()
    #
    # return reader

    smilies = []
    frontier = set()
    with open(filename, "r") as csvfile:
        if header:
            csvfile.readline()

        reader = csv.reader(csvfile, delimiter=delimiter)

        for row in reader:
            stripped_row = "".join(row).strip()

            if not stripped_row or stripped_row.startswith("#"):
                continue

            smiles = row[columns].strip()
            orig_smiles = smiles

            if actions:
                for action in actions:
                    if callable(action) and smiles:
                        smiles = action(orig_smiles)

            if not smiles:
                continue

            if smiles:  # SMILES transformation may fail
                if isinstance(smiles, list):
                    smiles = tuple(smiles)

                if (not remove_duplicates) or (not smiles in frontier):
                    smilies.append(smiles)
                    frontier.add(smiles)

    return smilies
