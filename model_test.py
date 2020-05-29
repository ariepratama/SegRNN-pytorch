from model import FrameIdentificationRNN, Param, configuration
from unittest.mock import Mock, MagicMock
import torch
from torchtext.vocab import FastText

TEXT_EMBEDDING = FastText('simple')

device = 'cpu'


vocdict_size = 3
postags_size = 3
ludict_size = 3
lpdict_size = 3
framedict_size = 3
p = Param(**{
    'vocdict_size': vocdict_size,
    'postdict_size': postags_size,
    'ludict_size': ludict_size,
    'lpdict_size': lpdict_size,
    'framedict_size': framedict_size
})

def _mock_voc_dict(values_map: dict = {0: 'he', 1: 'she'}):
    def _getstr(token_index: int):
        return values_map.get(token_index, '<unk>')

    vocdict_mock = Mock()
    vocdict_mock.getstr = _getstr
    return vocdict_mock


def test__tokens_to_vec():
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)
    token_inputs = torch.tensor([0, 1])
    token_embedding = mdl._tokens_to_vec(torch.tensor([0, 1]))
    assert token_embedding.size()[0] == token_inputs.size()[0]
    assert token_embedding.size()[1] == configuration['token_dim']


def test__postags_to_vec():
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)
    postag_inputs = torch.tensor([0, 1])
    postag_embedding = mdl._postags_to_vec(postag_inputs)
    assert postag_embedding.size()[0] == postag_inputs.size()[0]
    assert postag_embedding.size()[1] == configuration['pos_dim']


def test__tokens_and_postags_to_features():
    vocdict_mock = _mock_voc_dict()
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, vocdict_mock, p, device)
    token_inputs = torch.tensor([0, 1])
    postag_inputs = torch.tensor([0, 1])
    features = mdl._tokens_and_postags_to_features(
        token_inputs,
        postag_inputs
    )

    assert features.size()[0] == token_inputs.size()[0]
    assert features.size()[1] == p.lstmindim


def test__target_embeddings():
    targetpositions = [1]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)

    assert target_embeddings.size()[0] == len(targetpositions)
    assert target_embeddings.size()[1] == 1
    assert target_embeddings.size()[2] == 2 * features.size()[1]


def test__target_vec():
    targetpositions = [1]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)
    target_vec = mdl._target_vec(target_embeddings)
    print(target_vec)


def test__joint_embedding():
    targetpositions = [0, 1]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)
    target_vec = mdl._target_vec(target_embeddings)
    joint_embedding = mdl._joint_embedding(target_vec, torch.tensor([0, 0]), torch.tensor([0, 0]))
    print(joint_embedding)


def test_forward1d():
    targetpositions = [0]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)
    res = mdl(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0]),
        torch.tensor([0]),
        targetpositions
    )
    print(res)
    print(res.size())
    assert res.size()[1] == 3


def test_forward():
    targetpositions = [0, 1]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p, device)
    res = mdl(
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        targetpositions
    )
    print(res)
    print(res.size())
    assert res.size()[1] == 3


