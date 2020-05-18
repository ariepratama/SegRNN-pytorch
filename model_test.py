from model import FrameIdentificationRNN, Param, configuration
from unittest.mock import Mock, MagicMock
import torch
from torchtext.vocab import FastText
TEXT_EMBEDDING = FastText('simple')


def _mock_voc_dict(values_map: dict = {0: 'he', 1: 'she'}):
    def _getstr(token_index: int):
        return values_map.get(token_index, '<unk>')

    vocdict_mock = Mock()
    vocdict_mock.getstr = _getstr
    return vocdict_mock


def test__tokens_to_vec():
    vocdict_size = 2
    p = Param(**{
        'vocdict_size': vocdict_size
    })
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p)
    token_embedding = mdl._tokens_to_vec(torch.tensor([0, 1]))
    assert token_embedding.size()[0] == 2
    assert token_embedding.size()[1] == configuration['token_dim']


def test__postags_to_vec():
    postags_size = 2
    p = Param(**{
        'postdict_size': postags_size
    })
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p)
    postag_embedding = mdl._postags_to_vec(torch.tensor([0, 1]))
    assert postag_embedding.size()[0] == 2
    assert postag_embedding.size()[1] == configuration['pos_dim']


def test__tokens_and_postags_to_features():
    vocdict_size = 2
    postags_size = 2
    p = Param(**{
        'vocdict_size': vocdict_size,
        'postdict_size': postags_size
    })
    vocdict_mock = _mock_voc_dict()
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, vocdict_mock, p)
    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    assert features.size()[0] == p.lstmindim
    assert features.size()[1] == vocdict_size


def test__target_embeddings():
    vocdict_size = 2
    postags_size = 2
    p = Param(**{
        'vocdict_size': vocdict_size,
        'postdict_size': postags_size
    })
    targetpositions = [1, 0]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)

    assert target_embeddings.size()[0] == len(targetpositions)
    assert target_embeddings.size()[1] == 2 * features.size()[0]


def test__target_vec():
    vocdict_size = 2
    postags_size = 2
    p = Param(**{
        'vocdict_size': vocdict_size,
        'postdict_size': postags_size
    })
    targetpositions = [1, 0]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)
    target_vec = mdl._target_vec(target_embeddings)
    print(target_vec)


def test__joint_embedding():
    vocdict_size = 2
    postags_size = 2
    ludict_size = 1
    lpdict_size = 1
    framedict_size = 1
    p = Param(**{
        'vocdict_size': vocdict_size,
        'postdict_size': postags_size,
        'ludict_size': ludict_size,
        'lpdict_size': lpdict_size,
        'framedict_size': framedict_size
    })
    targetpositions = [1, 0]
    mdl = FrameIdentificationRNN(TEXT_EMBEDDING, _mock_voc_dict(), p)

    features = mdl._tokens_and_postags_to_features(
        torch.tensor([0, 1]),
        torch.tensor([0, 1])
    )

    target_embeddings = mdl._target_embeddings(features, targetpositions)
    target_vec = mdl._target_vec(target_embeddings)
    joint_embedding = mdl._joint_embedding(target_vec, 0, 0, device='cpu')
    print(joint_embedding)

