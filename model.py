import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext.vocab import Vectors
from conll09 import FspDict

configuration = {
    'unk_prob': 0.1,
    'dropout_rate': 0.01,
    'token_dim': 100,
    'pos_dim': 100,
    'lu_dim': 100,
    'lu_pos_dim': 100,
    'lstm_input_dim': 100,
    'lstm_dim': 100,
    'lstm_depth': 2,
    'hidden_dim': 100,
    'use_dropout': False,
    'pretrained_embedding_dim': 300,  # as torchtext by default uses 300
    'num_epochs': 3,
    'patience': 25,
    'eval_after_every_epochs': 100,
    'dev_eval_epoch_frequency': 5}

UNK_PROB = configuration['unk_prob']
DROPOUT_RATE = configuration['dropout_rate']

TOKDIM = configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
INPDIM = TOKDIM + POSDIM

LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']

NUM_EPOCHS = configuration['num_epochs']
PATIENCE = configuration['patience']
EVAL_EVERY_EPOCH = configuration['eval_after_every_epochs']
DEV_EVAL_EPOCH = configuration['dev_eval_epoch_frequency'] * EVAL_EVERY_EPOCH

PRETRAINED_EMB_DIM = configuration['pretrained_embedding_dim']


class Param(object):
    def __init__(self, **kwargs):
        self.vocdict_size = kwargs.get('vocdict_size', 0)
        self.tokdim = kwargs.get('tokdim', TOKDIM)
        self.postdict_size = kwargs.get('postdict_size', 0)
        self.posdim = kwargs.get('posdim', POSDIM)
        self.ludict_size = kwargs.get('ludict_size', 0)
        self.ludim = kwargs.get('ludim', LUDIM)
        self.lpdict_size = kwargs.get('lpdict_size', 0)
        self.lpdim = kwargs.get('lpdim', LPDIM)
        self.hiddendim = kwargs.get('hiddendim', HIDDENDIM)
        self.lstmindim = kwargs.get('lstmindim', LSTMINPDIM)
        self.lstmdim = kwargs.get('lstmdim', LSTMDIM)
        self.inpdim = kwargs.get('indim', INPDIM)
        self.framedict_size = kwargs.get('framedict_size', 0)
        self.pretrained_dim = kwargs.get('pretrained_dim', PRETRAINED_EMB_DIM)
        self.lstmdepth = kwargs.get('lstmdepth', LSTMDEPTH)


class FrameIdentificationRNN(nn.Module):
    """
    Pytorch Implementation of https://github.com/clab/dynet/tree/master/examples/segmental-rnn
    """

    def __init__(self, pretrained_embedding_map: Vectors, vocab_dict: FspDict, param: Param, device: str):
        super().__init__()
        self.pretrained_embedding_map = pretrained_embedding_map
        self.param = param
        self.vocab_dict = vocab_dict
        self.device = device

        self.v_x = nn.Embedding(param.vocdict_size, param.tokdim)
        self.p_x = nn.Embedding(param.postdict_size, param.posdim)
        self.lu_x = nn.Embedding(param.ludict_size, param.ludim)
        self.lp_x = nn.Embedding(param.lpdict_size, param.lpdim)

        self.lin_i = nn.Linear(param.lstmindim, param.inpdim)
        self.e_x = nn.Embedding(param.vocdict_size, param.pretrained_dim)
        # embedding for unknown pretrained embedding
        self.u_x = nn.Parameter(torch.rand(1, param.pretrained_dim), requires_grad=False)

        self.lin_e = nn.Linear(param.pretrained_dim + param.inpdim, param.lstmindim)

        self.fw_x = nn.LSTM(param.lstmindim, param.lstmdim, param.lstmdepth, bidirectional=True)
        self.fw_x_hidden = (
            torch.rand(param.lstmdepth * 2, 1, param.lstmdim).to(self.device),
            torch.rand(param.lstmdepth * 2, 1, param.lstmdim).to(self.device)
        )

        self.tlstm = nn.LSTM(param.lstmindim * 2, param.lstmdim, param.lstmdepth)
        self.tlstm_hidden = (
            torch.rand(param.lstmdepth, 1, param.lstmdim).to(self.device),
            torch.rand(param.lstmdepth, 1, param.lstmdim).to(self.device)
        )

        self.lin_z = nn.Linear(param.lstmdim + param.ludim + param.lpdim, param.hiddendim)
        self.lin_f = nn.Linear(param.hiddendim, param.framedict_size)

    def forward(self, tokens: torch.Tensor, postags: torch.Tensor, lexical_units: torch.Tensor,
                lexical_unit_postags: torch.Tensor, targetpositions: list) -> torch.Tensor:
        """

        :param tokens: a sentence tokens sequence represented as Tensor of int
        :param postags: a sentence POStag sequence represented as Tensor of int
        :param lexical_units: lexical_unit, that have `id` as lexical_unit id / int and posid: Part of speech of that lexical unit
        :param lexical_unit_postags: lexical_unit, that have `id` as lexical_unit id / int and posid: Part of speech of that lexical unit
        :param targetpositions: list, in what index does the Frame should be identified
        :return:
        """
        features_vec = self._tokens_and_postags_to_features(tokens, postags)
        target_embeddings = self._target_embeddings(features_vec, targetpositions)
        target_vec = self._target_vec(target_embeddings)
        return self._joint_embedding(
            target_vec,
            lexical_units,
            lexical_unit_postags
        )

    def forward_as_df(self, tokens: torch.Tensor, postags: torch.Tensor, lexical_units: torch.Tensor,
                      lexical_unit_postags: torch.Tensor, targetpositions: list):
        import pandas as pd
        import numpy as np
        """

        :param tokens: a sentence tokens sequence represented as Tensor of int
        :param postags: a sentence POStag sequence represented as Tensor of int
        :param lexical_units: lexical_unit, that have `id` as lexical_unit id / int and posid: Part of speech of that lexical unit
        :param lexical_unit_postags: lexical_unit, that have `id` as lexical_unit id / int and posid: Part of speech of that lexical unit
        :param targetpositions: list, in what index does the Frame should be identified
        :return:
        """
        features_vec = self._tokens_and_postags_to_features(tokens, postags)
        target_embeddings = self._target_embeddings(features_vec, targetpositions)
        target_vec = self._target_vec(target_embeddings)
        joint_embedding = self._joint_embedding(
            target_vec,
            lexical_units,
            lexical_unit_postags
        )

        return pd.DataFrame(np.array([
            torch.mean(features_vec).item(), torch.sum(features_vec).item(),
            torch.mean(target_embeddings).item(), torch.sum(target_embeddings).item(),
            torch.mean(target_vec).item(), torch.sum(target_vec).item(),
            torch.mean(joint_embedding).item(), torch.sum(joint_embedding).item()
        ]).reshape(1, -1), columns=[
            'mean_feature_vec',
            'sum_feature_vec',
            'mean_target_embeddings',
            'sum_target_embeddings',
            'mean_target_vec',
            'sum_target_vec',
            'mean_joint_embedding',
            'sum_joint_embedding'
        ])

    def _tokens_to_vec(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        return embedding for tokens
        :param tokens:
        :return:
        """
        return self.v_x(tokens)

    def _get_token_pretrained_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """

        :param tokens:
        :return:
        """
        placeholders = torch.zeros(tokens.size()[0], self.param.pretrained_dim).to(self.device)
        for i, token in enumerate(tokens):
            placeholders[i] = self.pretrained_embedding_map[
                self.vocab_dict.getstr(token.cpu().item())
            ].to(self.device)
        return placeholders

    def _postags_to_vec(self, postags: torch.Tensor) -> torch.Tensor:
        """
        return embedding for postags
        :param postags:
        :return:
        """
        return self.p_x(postags)

    def _tokens_and_postags_to_features(self, tokens: torch.Tensor, postags: torch.Tensor) -> torch.Tensor:
        """
        return intermediary feature before going to lstm for forward and backward
        :param tokens_vec:
        :param postags_vec:
        :return:
        """

        features = torch.cat([
            self._tokens_to_vec(tokens),
            self._postags_to_vec(postags),
            self._get_token_pretrained_embedding(tokens)
        ], dim=1)
        features = self.lin_e(features)
        features = F.relu(features)
        return features

    def _target_embeddings(self, feature_vec: torch.Tensor, targetpositions: list) -> torch.Tensor:
        """

        :param feature_vec: (lstm_in_dim, vocab_size)
        :param targetpositions:
        :return:
        """
        feature_vec = feature_vec
        feature_vec = feature_vec.view(
            feature_vec.size()[0],
            1,  # batch num, assuming this to be 1
            feature_vec.size()[1]
        )
        # TODO reproduce dropout
        # if USE_DROPOUT and trainmode:
        #     builders[0].set_dropout(DROPOUT_RATE)
        #     builders[1].set_dropout(DROPOUT_RATE)

        forward_feature, _ = self.fw_x(feature_vec)
        # only take vector in frame position
        forward_feature = forward_feature[targetpositions]
        return forward_feature

    def _target_vec(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        x, _ = self.tlstm(target_embeddings)

        return x

    def _joint_embedding(self, target_vec: torch.Tensor, lu_id: torch.Tensor, posid: torch.Tensor) -> torch.Tensor:
        # TODO reproduce this
        # if USE_HIER and lexunit.id in relatedlus:
        #     lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        # else:
        #     lu_vec = lu_x[lexunit.id]
        # if len(target_vec.size()) == 1:
        #     target_vec = target_vec.view(1, -1)
        target_vec = target_vec.squeeze(1)

        fbemb_i = torch.cat([
            target_vec,
            self.lu_x(lu_id),
            self.lp_x(posid)
        ], dim=1)

        # TODO reproduce this dropout
        # if trainmode and USE_DROPOUT:
        #     f_i = dropout(f_i, DROPOUT_RATE)

        x = self.lin_z(fbemb_i)
        # x = F.relu(x)
        x = self.lin_f(x)
        # x = F.relu(x)
        return x
        # return self.w_f.mm(x) + self.b_f


class FrameTargetIdentificationRNNParam(object):
    def __init__(self, **kwargs):
        self.input_size = kwargs.get('input_size', 0)
        self.token_dim = kwargs.get('token_dim', 100)
        self.pretrained_dim = kwargs.get('pretrained_dim', 300)
        self.postag_size = kwargs.get('postag_size', 0)
        self.postag_dim = kwargs.get('postag_dim', 100)
        self.lemma_size = kwargs.get('lemma_size', 0)
        self.lemma_dim = kwargs.get('lemma_dim', 100)
        self.bilstm_input_size = kwargs.get('bilstm_input_size', 100)
        self.bilstm_hidden_size = kwargs.get('bilstm_hidden_size', 100)
        self.bilstm_layer_size = kwargs.get('bilstm_layer_size', 2)
        self.output_size = kwargs.get('output_size', 0)


class FrameTargetIdentificationRNN(nn.Module):
    def __init__(self, pretrained_embedding: Vectors, model_param: FrameTargetIdentificationRNNParam):
        super().__init__()
        self.token_embedding = nn.Embedding(model_param.input_size, model_param.token_dim)
        self.postag_embedding = nn.Embedding(model_param.postag_size, model_param.postag_dim)
        self.lemma_embedding = nn.Embedding(model_param.lemma_size, model_param.lemma_dim)
        self.pretrained_embedding = pretrained_embedding

        self.lin1 = nn.Linear(
            (model_param.token_dim +
             model_param.postag_dim +
             model_param.lemma_dim +
             model_param.pretrained_dim),
            model_param.bilstm_input_size
        )
        self.bilstm = nn.LSTM(
            model_param.bilstm_input_size,
            model_param.bilstm_hidden_size
        )
        self.lin2 = nn.Linear(
            model_param.bilstm_hidden_size,
            model_param.output_size
        )

    def forward(self, tokens: torch.Tensor, postags: torch.Tensor, lemmas: torch.Tensor):
        tokens_x = self.token_embedding(tokens)
        postags_x = self.postag_embedding(postags)
        lemmas_x = self.lemma_embedding(lemmas)
        pretrained_x = torch.zeros(tokens.shape[0], self.pretrained_embedding.dim)
        for i, token in enumerate(tokens):
            pretrained_x[i] = self.pretrained_embedding[token]
        x = torch.cat([tokens_x, postags_x, lemmas_x, pretrained_x], dim=1)
        x = self.lin1(x)
        x = F.relu(x)
        x, _ = self.bilstm(x.view(x.shape[0], 1, x.shape[1]))
        x = F.relu(x)
        x = self.lin2(x)

        return x


class ArgumentIdentificationBaseEmbeddingParam(object):
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 0)
        self.vocab_embedding_dim = kwargs.get('vocab_embedding_dim', 0)
        self.postag_input_dim = kwargs.get('postag_input_dim', 0)
        self.postag_dim = kwargs.get('postag_dim', 0)
        self.bilstm_input_dim = kwargs.get('bilstm_input_dim', 0)
        self.bilstm_hidden_dim = kwargs.get('bilstm_hidden_dim', 0)
        self.bilstm_n_layers = kwargs.get('bilstm_n_layers', 0)


class ArgumentIdentificationBaseEmbedding(nn.Module):
    def __init__(self, model_param: ArgumentIdentificationBaseEmbeddingParam):
        super().__init__()
        self.vocab_embedding = nn.Embedding(model_param.input_dim, model_param.vocab_embedding_dim)
        self.postag_embedding = nn.Embedding(model_param.postag_input_dim, model_param.postag_dim)
        self.lin1 = nn.Linear(
            model_param.input_dim + model_param.postag_input_dim + model_param.input_dim,
            model_param.bilstm_input_dim
        )
        self.bilstm = nn.LSTM(
            model_param.bilstm_input_dim,
            model_param.bilstm_hidden_dim,
            model_param.bilstm_n_layers
        )
        self.lin2 = nn.Linear(
            model_param.bilstm_hidden_dim,
            model_param.bilstm_input_dim
        )

    def forward(self, tokens: torch.Tensor, postags: torch.Tensor, distances_from_target_frame: torch.Tensor):
        vocab_embedded = self.vocab_embedding(tokens)
        postag_embedded = self.postag_embedding(postags)
        x = torch.cat([
            vocab_embedded,
            postag_embedded,
            distances_from_target_frame
        ], dim=1)
        x = F.relu(x)
        x, _ = self.bilstm(x.view(x.shape[0], 1, x.shape[1]))
        x = F.relu(x.squeeze(1))

        return x


class ArgumentIdentificationFrameEmbeddingParam(object):
    def __init__(self, **kwargs):
        self.lu_size = kwargs.get('lu_size', 0)
        self.lu_embedding_size = kwargs.get('lu_embedding_size', 0)
        self.lu_postag_size = kwargs.get('lu_postag_size', 0)
        self.lu_postag_embedding_size = kwargs.get('lu_postag_embedding_size', 0)
        self.joint_embedding_size = kwargs.get('joint_embedding_size', 0)
        self.lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 0)
        self.lstm_n_layers = kwargs.get('lstm_n_layers', 0)
        self.frame_size = kwargs.get('frame_size', 0)
        self.frame_embedding_size = kwargs.get('frame_embedding_size', 0)
        self.ctx_size = kwargs.get('ctx_size', 0)


class ArgumentIdentificationFrameEmbedding(nn.Module):
    def __init__(self, model_param: ArgumentIdentificationFrameEmbeddingParam):
        super().__init__()

        self.lu_embedding = nn.Embedding(
            model_param.lu_size,
            model_param.lu_embedding_size
        )
        self.lu_postag_embedding = nn.Embedding(
            model_param.lu_postag_size,
            model_param.lu_postag_embedding_size
        )
        self.frame_embedding = nn.Embedding(
            model_param.frame_size,
            model_param.frame_embedding_size
        )
        self.lstm = nn.LSTM(
            model_param.joint_embedding_size,
            model_param.lstm_hidden_dim,
            model_param.lstm_n_layers
        )
        self.ctx_lstm = nn.LSTM(
            model_param.ctx_size,
            model_param.lstm_hidden_dim,
            model_param.lstm_n_layers
        )

    def forward(self, joint_embedding: torch.Tensor, ctx: torch.Tensor,
                lu: torch.Tensor, lu_postag: torch.Tensor, frame: torch.Tensor):
        x_joint = self.lstm(joint_embedding.view(joint_embedding.shape[0], 1, joint_embedding.shape[1]))
        x_ctx = self.ctx_lstm(ctx)
        x_lu = self.lu_embedding(lu)
        x_lu_postag = self.lu_postag_embedding(lu_postag)
        x_frame = self.frame_embedding(frame)

        return torch.cat([
            x_lu,
            x_lu_postag,
            x_frame,
            x_joint,
            x_ctx
        ], dim=1)
