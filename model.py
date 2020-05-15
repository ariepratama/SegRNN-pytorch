import torch.nn as nn
import torch
import torch.nn.functional as F

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
    'pretrained_embedding_dim': 100,
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

    def __init__(self, pretrained_embedding_map, param: Param):
        super().__init__()
        self.pretrained_embedding_map = pretrained_embedding_map
        # TODO FIX THIS
        # PRETRAINED_DIM = len(list(self.pretrained_embedding_map.values()))

        self.v_x = nn.Embedding(param.vocdict_size, param.tokdim)
        self.p_x = nn.Embedding(param.postdict_size, param.posdim)
        self.lu_x = nn.Embedding(param.ludict_size, param.ludim)
        self.lp_x = nn.Embedding(param.lpdict_size, param.lpdim)

        self.w_i = nn.Parameter(
            torch.rand(param.lstmindim, param.inpdim), requires_grad=True)
        self.b_i = nn.Parameter(
            torch.rand(param.lstmindim, 1), requires_grad=True)

        self.w_z = nn.Parameter(
            torch.rand(param.hiddendim, param.lstmdim + param.ludim + param.lpdim), requires_grad=True)
        self.b_z = nn.Parameter(
            torch.rand(param.hiddendim, 1), requires_grad=True)

        self.w_f = nn.Parameter(
            torch.rand(param.framedict_size, param.hiddendim), requires_grad=True)
        self.b_f = nn.Parameter(
            torch.rand(param.framedict_size, 1), requires_grad=True)

        self.e_x = nn.Embedding(param.vocdict_size, param.pretrained_dim)
        # embedding for unknown pretrained embedding
        self.u_x = nn.Parameter(torch.rand(1, param.pretrained_dim), requires_grad=True)

        self.w_e = nn.Parameter(
            torch.rand(param.lstmindim, param.pretrained_dim + param.inpdim), requires_grad=True)
        self.b_e = nn.Parameter(
            torch.rand(param.lstmindim, 1), requires_grad=True)

        self.fw_x = nn.LSTM(param.lstmindim, param.lstmdim, param.lstmdepth, bidirectional=True)
        self.fw_x_hidden = torch.rand(param.lstmdim, param.lstmdepth)

        self.tlstm = nn.LSTM(param.lstmindim * 2, param.lstmdim, param.lstmdepth)
        self.tlstm_hidden = torch.rand(param.lstmdim, param.lstmdepth)

    def forward(self, tokens: torch.Tensor, postags: torch.Tensor, lexical_unit, targetpositions: list, device: str) -> torch.Tensor:
        """

        :param tokens: a sentence tokens sequence represented as Tensor of int
        :param postags: a sentence POStag sequence represented as Tensor of int
        :param lexical_unit: lexical_unit, that have `id` as lexical_unit id / int and posid: Part of speech of that lexical unit
        :param targetpositions: list, in what index does the Frame should be identified
        :return:
        """
        tokens_vec = self._tokens_to_vec(tokens)
        postags_vec = self._postags_to_vec(postags)
        features_vec = self._tokens_and_postags_to_features(tokens_vec, postags_vec)
        target_embeddings = self._target_embeddings(features_vec, targetpositions)
        target_vec = self._target_vec(target_embeddings)
        return self._joint_embedding(
            target_vec,
            lexical_unit.id,
            lexical_unit.posid,
            device
        )

    def _tokens_to_vec(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        return embedding for tokens
        :param tokens:
        :return:
        """
        return self.v_x(tokens)

    def _postags_to_vec(self, postags: torch.Tensor) -> torch.Tensor:
        """
        return embedding for postags
        :param postags:
        :return:
        """
        return self.p_x(postags)

    def _tokens_and_postags_to_features(self, tokens_vec: torch.Tensor, postags_vec: torch.Tensor) -> torch.Tensor:
        """
        return intermediary feature before going to lstm for forward and backward
        :param tokens_vec:
        :param postags_vec:
        :return:
        """
        features = torch.cat([tokens_vec, postags_vec, self.u_x.repeat(tokens_vec.size()[0], 1)], dim=1)
        return F.relu(self.w_e.mm(features.T) + self.b_e)

    def _target_embeddings(self, feature_vec: torch.Tensor, targetpositions: list) -> torch.Tensor:
        """

        :param feature_vec: (lstm_in_dim, vocab_size)
        :param targetpositions:
        :return:
        """
        feature_vec = feature_vec.T
        feature_vec = feature_vec.view(
            feature_vec.size()[0],
            1, #batch num, assuming this to be 1
            feature_vec.size()[1]
        )
        # TODO reproduce dropout
        # if USE_DROPOUT and trainmode:
        #     builders[0].set_dropout(DROPOUT_RATE)
        #     builders[1].set_dropout(DROPOUT_RATE)

        # TODO this can be simplified using bidirectional = True
        forward_feature, _ = self.fw_x(feature_vec)
        # backward_feature, _ = self.bw_x(torch.flip(feature_vec, [2]))

        # target_embeddings = torch.zeros(
        #     len(targetpositions),
        #     forward_feature.size()[1] * 2,
        #     forward_feature.size()[2]
        # )
        # sentlen = forward_feature.size()[0]
        #
        # j = 0
        # for targetidx in targetpositions:
        #     target_embeddings[j] = torch.cat((
        #         forward_feature[targetidx], backward_feature[sentlen - targetidx - 1]
        #     ))
        #     j += 1
        return forward_feature.squeeze()

    def _target_vec(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        target_embeddings = target_embeddings.view(
            target_embeddings.size()[0],
            1,
            target_embeddings.size()[1]
        )
        x, h = self.tlstm(target_embeddings)
        return x[-1]

    def _joint_embedding(self, target_vec: torch.Tensor, lu_id: int, posid: int, device: str) -> torch.Tensor:
        # TODO reproduce this
        # if USE_HIER and lexunit.id in relatedlus:
        #     lu_vec = esum([lu_x[luid] for luid in relatedlus[lexunit.id]])
        # else:
        #     lu_vec = lu_x[lexunit.id]
        target_vec = target_vec.squeeze()
        lu_vec = self.lu_x(torch.tensor(lu_id).to(device))

        fbemb_i = torch.cat([target_vec, lu_vec, self.lp_x(torch.tensor(posid).to(device))])

        # TODO reproduce this dropout
        # if trainmode and USE_DROPOUT:
        #     f_i = dropout(f_i, DROPOUT_RATE)

        # f_i = w_f * rectify(w_z * fbemb_i + b_z) + b_f
        x = F.relu(self.w_z.mm(fbemb_i.unsqueeze(1)) + self.b_z)
        return self.w_f.mm(x) + self.b_f

    def _get_valid_frames(self, lexical_unit):
        return list(self.lexical_unit_frame_map[lexical_unit.id])

    def _chosen_frame(self, valid_frames):
        return valid_frames[0]
