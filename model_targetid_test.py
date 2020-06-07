from model import FrameTargetIdentificationRNN, FrameTargetIdentificationParam, configuration
from unittest.mock import Mock, MagicMock
import torch
from torchtext.vocab import FastText

TEXT_EMBEDDING = FastText('simple')

device = 'cpu'

model_param = FrameTargetIdentificationParam(**dict(
    input_size=2,
    postag_size=3,
    lemma_size=4,
    output_size=2,
    pretrained_dim=TEXT_EMBEDDING.dim
))
tokens = torch.tensor([0, 1])
postags = torch.tensor([1, 2])
lemmas = torch.tensor([2, 3])


def lin1(model: FrameTargetIdentificationRNN, tokens, postags, lemmas):
    tokens_x = model.token_embedding(tokens)
    postags_x = model.postag_embedding(postags)
    lemmas_x = model.lemma_embedding(lemmas)
    pretrained_x = torch.zeros(tokens.shape[0], model.pretrained_embedding.dim)
    for i, token in enumerate(tokens):
        pretrained_x[i] = model.pretrained_embedding[token]

    x = torch.cat([tokens_x, postags_x, lemmas_x, pretrained_x], dim=1)
    return model.lin1(x)


def test_lin1():
    model = FrameTargetIdentificationRNN(TEXT_EMBEDDING, model_param)
    x = lin1(model, tokens, postags, lemmas)

    assert x.shape[0] == len(tokens)
    assert x.shape[1] == model_param.bilstm_input_size


def test_bilstm():
    model = FrameTargetIdentificationRNN(TEXT_EMBEDDING, model_param)
    x = lin1(model, tokens, postags, lemmas)

    x, _ = model.bilstm(x.view(x.shape[0], 1, x.shape[1]))

    assert len(x.shape) == 3
    assert x.shape[0] == model_param.bilstm_layer_size
    assert x.shape[1] == model_param.bilstm_hidden_size


def test_with_sentence_input():
    model = FrameTargetIdentificationRNN(TEXT_EMBEDDING, model_param)

    model.forward()