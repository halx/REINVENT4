"""Create a Lbinvent model from a list of SMILES strings"""

from __future__ import annotations
import uuid

from reinvent.models.libinvent.models.model import DecoratorModel
from reinvent.models.libinvent.models.vocabulary import DecoratorVocabulary
from reinvent.models.libinvent.models.decorator import Decorator
from reinvent.chemistry.file_reader import FileReader
from reinvent.models.meta_data2 import ModelMetaDataV2


def create_model(
    num_layers: int,
    layer_size: int,
    dropout: float,
    max_sequence_length: int,
    input_smiles_path: str,
    output_model_path: str,
    stereo: bool,
):
    """Create a Lbinvent model from scratch

    Learn the vocabulary from SMILES.

    :returns: a new Libinvent model
    """

    reader = FileReader([], None)

    # build vocabulary
    scaffolds, decorators = zip(
        *reader.read_library_design_data_file(input_smiles_path, num_fields=2)
    )

    vocabulary = DecoratorVocabulary.from_lists(scaffolds, decorators)

    encoder_config = {
        "num_layers": num_layers,
        "num_dimensions": layer_size,
        "dropout": dropout,
        "vocabulary_size": vocabulary.len_scaffold(),  # FIXME: is this right?
    }

    # FIXME: is len right?
    decoder_config = {**encoder_config, "vocabulary_size": vocabulary.len_decoration()}

    network = Decorator(encoder_config, decoder_config)

    metadata = ModelMetaDataV2(
        hash_id=None,
        model_id=str(uuid.uuid4()),
        model_architecture=DecoratorModel._model_type,
        supports_stereochemistry=stereo,
    )

    model = DecoratorModel(
        vocabulary=vocabulary,
        decorator=network,
        meta_data=metadata,
        max_sequence_length=max_sequence_length,
    )

    model.save(output_model_path)

    return model


if __name__ == "__main__":
    import sys

    input_smiles_path = sys.argv[1]
    output_model_path = sys.argv[2]
    stereo = bool(sys.argv[3])

    create_model(
        num_layers=3,
        layer_size=512,
        dropout=0.0,
        max_sequence_length=256,
        input_smiles_path=input_smiles_path,
        output_model_path=output_model_path,
        stereo=stereo,
    )
