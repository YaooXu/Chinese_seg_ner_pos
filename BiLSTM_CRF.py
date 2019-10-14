import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

STOP_label = '<stop>'
START_label = '<start>'
PAD_label = '<pad>'

def prepare_sequence(seq, word2idx):
    idxs = [word2idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec: torch.tensor, dim=0):
    r"""
    计算向量某个维度上的logsumexp
    :param vec: Tensor
    :param dim: 在哪个维度计算log_sum_exp, 默认为0
    :return: Tensor (1, vec.shape[1])
        分别表示vec每一列上的log_sum_exp
    """

    max_score = vec.max(dim=dim, keepdim=True)[0]
    return max_score.squeeze(dim=1) + torch.log(torch.sum(torch.exp(vec - max_score), dim=dim))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, label2idx, embedding_dim, hidden_size, num_layers, dropout_ratio):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.labelset_size = len(label2idx)
        self.embedding_dim = embedding_dim
        self.label2idx = label2idx
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 非BatchFirst在实际其实更方便。。。
        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # 把output转化为label
        self.output2label = nn.Linear(hidden_size, self.labelset_size)

        # 标签的转移得分
        # transitons[i, j] 表示 从 i 转移到 j 的得分
        self.transitions = nn.Parameter(
            torch.randn(self.labelset_size, self.labelset_size, requires_grad=True))

        # 不可能从STOP转移到其他标签，也不可能从其他标签转移到START
        # TODO: 为啥非要加.detach....?
        self.transitions.detach()[label2idx[STOP_label], :] = -10000
        self.transitions.detach()[:, label2idx[START_label]] = -10000

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.hidden = None  # self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        r"""
        初始化隐藏层参数
        :param batch_size:  batch_size
        :return:
        """
        return (torch.randn(2, batch_size, self.hidden_size // 2),
                torch.randn(2, batch_size, self.hidden_size // 2))

    def _get_lstm_features(self, sentences):
        '''
        得到序列的特征
        :param sentences: tensor [batch_size, length]
        :return: feats tensor [batch_size, length, labelset_size]
        '''

        self.hidden = self.init_hidden(sentences.shape[0])
        # [batch_size, length] -> [batch_size, length, dim]
        sentences_embeddings = self.embeddings(sentences)
        sentences_embeddings = self.dropout1(sentences_embeddings)

        # outputs [batch_size, length, hidden_size]
        outputs, self.hidden = self.LSTM(sentences_embeddings, self.hidden)
        outputs = self.dropout2(outputs)

        feats = self.output2label(outputs)
        return feats

    def _forward_all_logsumexp(self, feats, masks):
        r"""
        计算所有可能路径的log_sum_exp
        :param feats: tensor [batch_size, length, labelset_size]
            LSTM传过来的特征向量
        :param masks: tensor [batch_size, length]
        :return: terminal_score: tensor [batch_size]
        """
        batch_size = feats.shape[0]
        length = feats.shape[1]

        # 到当前单词，且状态为i的所有路径的log_sum_exp
        # TODO : 所有tensor device一样
        dp = torch.full((batch_size, self.labelset_size), -10000.)
        # START_label has all of the score.
        dp[:, self.label2idx[START_label]] = 0.
        for i in range(length):
            # [batch_size, labelset_size]
            feat = feats[:, i, :]
            # [batch_size] -> [batch_size, 1]
            mask = masks[:, i].unsqueeze(dim=1)
            # [labelset_size, batch_size, labelset_size]
            tmp = dp.transpose(0, 1).unsqueeze(dim=2) + \
                  feat.unsqueeze(dim=0) + \
                  self.transitions.unsqueeze(dim=1)
            # [labelset_size, batch_size, labelset_size] -> [batch_size, labelset_size]
            tmp = log_sum_exp(tmp, dim=0)
            # mask为1的值更新，为0的不再更新
            dp.masked_scatter_(mask, tmp.masked_select(mask))

        terminal_var = dp + self.transitions[:, self.label2idx[STOP_label]]
        terminal_score = log_sum_exp(terminal_var, dim=1)

        return terminal_score

    def _get_gold_score(self, feats, labels, masks):
        '''
        计算出所提供的正确路径得分数
        :param feats: tensor [batch_size, length, labelset_size]
            LSTM传过来的特征向量
        :param labels: tensor [batch_size, length]
            每个序列正确的路径, 已经加了start
        :param masks: tensor [batch_size, length]
        :return:
           scores: tensor [batch_size]
        '''
        batch_size = feats.shape[0]

        scores = torch.zeros((batch_size))
        start = torch.full([batch_size, 1], self.label2idx[START_label], dtype=torch.long)

        # [batch_size, length + 1]
        labels = torch.cat(
            [start, labels], dim=1
        )

        for i in range(feats.shape[1]):
            # [batch_size, labelset_size]
            feat = feats[:, i, :]
            # [batch_size]
            mask = masks[:, i]
            tmp = scores + self.transitions[labels[:, i], labels[:, i + 1]] + feat[range(batch_size), labels[:, i + 1]]
            # mask为1的值更新，为0的不再更新
            scores.masked_scatter_(mask, tmp.masked_select(mask))

        scores = scores + self.transitions[labels[:, -1], self.label2idx[STOP_label]]

        return scores

    def _viterbi_decode(self, feats, masks):
        r'''
        使用维特比算法进行解码，找到最可能的序列结果

        :param feats: tensor [batch_size, length, labelset_size]
            LSTM传过来的特征向量
        :param masks: tensor [batch_size, length]
        :return: best_scores tensor [batch_size]
                best_paths list made of tensor [batch_size] with length=length
        '''
        batch_size = feats.shape[0]
        # 记录每个节点由哪个父节点转移过来

        parents = []
        # 到当前单词，且状态为i的所有路径中log_sum_exp最大的值
        dp = torch.full((batch_size, self.labelset_size), -10000.)
        # START_label has all of the score.
        dp[:, self.label2idx[START_label]] = 0.
        for i in range(feats.shape[1]):
            # [batch_size, labelset_size]
            feat = feats[:, i, :]
            # [batch_size] -> [batch_size, 1]
            mask = masks[:, i].unsqueeze(dim=1)
            # [labelset_size, batch_size, labelset_size]
            # TODO: 搞清楚这些维数！！
            tmp = dp.transpose(0, 1).unsqueeze(dim=2) + \
                  feat.unsqueeze(dim=0) + \
                  self.transitions.unsqueeze(dim=1)
            max_scores, best_choose = tmp.max(dim=0)
            # 添加路径信息，[batch_size, labelset_size]
            parents.append(best_choose)

            # 由于只保留了一条路径，可以省去log_sum_exp过程
            dp.masked_scatter_(mask, max_scores.masked_select(mask))

        # [batch_size, labelset_size]
        terminal_var = dp + self.transitions[:, self.label2idx[STOP_label]]

        # [batch_size]
        best_scores, best_path_labels = terminal_var.max(dim=1)

        best_paths = [best_path_labels]
        for parent in reversed(parents):
            best_path_labels = parent[range(parent.shape[0]), best_paths[-1]]
            best_paths.append(best_path_labels)

        best_paths.pop()
        best_paths.reverse()

        # 转化为 [batch_size, length]
        best_paths = torch.stack(best_paths).transpose(0, 1)
        return best_scores, best_paths

    def neg_log_likelihood(self, sentences, labels, masks):
        r"""
        计算正确路径的负对数似然概率

        :param sentences: tensor [batch_size, length, dim]
        :param labels: tensor [batch_size, length]
            正确的label序列
        :param masks:tensor [batch_size, length]
        :return: FloatTensor
        """

        feats = self._get_lstm_features(sentences)

        forward_score = self._forward_all_logsumexp(feats, masks)
        gold_score = self._get_gold_score(feats, labels, masks)

        # print('forward_score: ', forward_score)
        # print('gold_score   :', gold_score)

        return torch.mean(forward_score - gold_score)

    def forward(self, sentences, masks):  # dont confuse this with _forward_alg above.
        r"""
        预测数据的最可能序列以及得分

        :param sentences: tensor [batch_size, length]
        :return:
            scores: tensor [batch_size]
            paths: list made of tensor [batch_size] with length=length
        """
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(sentences)
        # Find the best path, given the features.
        scores, paths = self._viterbi_decode(feats, masks)
        return scores, paths


class CRFDataset(Dataset):
    """Dataset Class for word-level model

    args:
        data_tensor (ins_num, seq_length): words
        label_tensor (ins_num, seq_length): labels
        mask_tensor (ins_num, seq_length): padding masks
    """

    def __init__(self, data_tensor, label_tensor, mask_tensor):
        assert data_tensor.size(0) == label_tensor.size(0)
        assert data_tensor.size(0) == mask_tensor.size(0)
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.mask_tensor = mask_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index], self.mask_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
