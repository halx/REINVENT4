import pytest

import torch
import numpy as np

from reinvent.runmodes.RL.memories.inception import Inception

SMILIES = (
    "CCN(CC(=O)Nc1ccc(S(N)(=O)=O)cc1)CC(=O)Nc1ccccc1OC",
    "CC(c1nnnn1C)N(C)C1CCN(CCn2cccn2)CC1",
    "Nc1nc(NCCN2CCC(CO)CC2)cc(C2CC(N)C2)n1",
    "CCc1nc(CN2CCN(Cc3ccc4c(c3)OCO4)C(=O)C2)n[nH]1",
    "CN(C)CCNC(=O)Cn1cc(NC(=O)C2CCC2)cn1",
    "COc1cc(F)c(C(=O)NCCNC2CCCCCC2)cc1OC",
    "O=c1c2ccccc2n2c(SCc3cccc(Oc4ccccc4)c3)nnc2n1-c1ccccc1",
    "COC(=O)C(c1ccccc1Cl)N1CCN(C(=O)c2ccc(Cl)cc2)CC1",
    "Cc1nn(CC(=O)c2c(-c3ccccc3)[nH]c3ccccc23)c(C)c1[N+](=O)[O-]",
    "CC(C)N1CCN(c2ccc(CCNCCCN3CCCC3)cc2)CC1",
    "COCCn1cnnc1CCNC(=O)CN1CCCC(C)C1",
    "O=C(c1cnc[nH]1)N1CCC2(CC1)CN(CC1CCOC1)CCO2",
    "Cc1nn(C)c(C)c1C1CCCN1CC(O)CN1CCOCC1",
    "CN(C)CC(NC(=O)c1ccc(=O)n(Cc2ccccc2Cl)c1)c1ccccc1",
    "Cc1cc(F)ccc1CC(=O)NCC1(O)CCCNC1",
    "O=C(NCCN1CCOCC1)Nc1cccc(OCc2cccc(F)c2)c1",
    "COc1ccc(CNCC(=O)Nc2c(C)cccc2C(C)C)cc1",
    "Cc1ccccc1NC(=O)Cc1nc(-c2cc(C)n(CC3COc4ccccc4O3)c2C)cs1",
    "Nc1ncc(CNCCOc2ccccn2)cn1",
    "C1CN(CCNC2CSCCSC2)CCN1",
)

SAMPLE_SIZE = 4
MEMORY_SIZE = len(SMILIES)


@pytest.fixture
def inception_memory():
    scores = torch.linspace(0.2, 0.9, 20)
    lls = torch.arange(1, 21, dtype=torch.float)

    inception = Inception(
        memory_size=MEMORY_SIZE,
        sample_size=SAMPLE_SIZE,
        seed_smilies=None,
        scoring_function=None,
        prior=None,
    )

    inception.add(SMILIES, scores, lls)

    yield inception


def test_inception_add(inception_memory):
    assert len(inception_memory.storage) == len(SMILIES)
    assert inception_memory._storage_smilies == set(SMILIES)

    smilies, _, _, _, _ = zip(*inception_memory.storage)

    assert list(smilies) == list(reversed(SMILIES))


def test_inception_sample(inception_memory):
    sampled, is_weight_max, is_weight_entropy, mean_drift = inception_memory.sample()

    assert sampled is not None
    assert len(sampled) == 5  # smilies, scores, prior_lls, agent_lls, relevance
    assert len(sampled[0]) == SAMPLE_SIZE
    assert len(sampled[1]) == SAMPLE_SIZE
    assert len(sampled[2]) == SAMPLE_SIZE
    assert len(sampled[3]) == SAMPLE_SIZE
    assert len(sampled[4]) == SAMPLE_SIZE

    # no agent provided — IS metrics should be None
    assert is_weight_max is None
    assert is_weight_entropy is None
    assert mean_drift is None


def test_inception_add_more_and_sample(inception_memory):
    smilies = (
        "CC1CCCCC1NC(=O)C(=O)NCCCN1CCOCC1",
        "CC(C)C(N)C(=O)N1CCN(c2ccc(CN(C)C)cn2)CC1",
        "O=CI(CN1CCCC(C(=O)Nc2ccccc2)C1)NCC(=O)Nc1ccc(F)c(F)c1",
    )
    scores = torch.tensor([0.17, 0.08, 0.12])
    lls = torch.tensor([-12.19, -5.13, -7.87])

    result = inception_memory(smilies, scores, lls)
    _smilies, _scores, _lls, _agent_lls, _relevance = result
    assert len(_smilies) == SAMPLE_SIZE
    assert len(_scores) == SAMPLE_SIZE
    assert len(_lls) == SAMPLE_SIZE

    _smilies, _scores, _lls, _agent_lls, _relevance = zip(*inception_memory.storage)
    assert len(_smilies) == MEMORY_SIZE
    assert len(_scores) == MEMORY_SIZE
    assert len(_lls) == MEMORY_SIZE
