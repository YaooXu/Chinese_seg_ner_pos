import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import *


def prepare_sequence(seq, word2idx):
    idxs = [word2idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class BiLSTM_CRF_S(nn.Module):
    def __init__(self, vocab_size, label2idx, embedding_dim, hidden_size, num_layers, dropout_ratio=0.3):
        super(BiLSTM_CRF_S, self).__init__()
        self.vocab_size = vocab_size
        self.labelset_size = len(label2idx)
        self.embedding_dim = embedding_dim
        self.label2idx = label2idx
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 非BatchFirst在实际其实更方便...
        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            # batch_first=True
        )

        # 把output转化为label
        self.output2label = nn.Linear(hidden_size, self.labelset_size)

        # 标签的转移得分
        # transitons[i, j] 表示 从 i 转移到 j 的得分
        self.transitions = nn.Parameter(
            torch.randn(self.labelset_size, self.labelset_size, requires_grad=True))

        # 不可能从STOP转移到其他标签，也不可能从其他标签转移到START
        # 必须要加detach
        self.transitions.detach()[label2idx[STOP_label], :] = -10000
        self.transitions.detach()[:, label2idx[START_label]] = -10000

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.hidden = None
        self.seq_length = None
        self.batch_size = None

    def init_uniform(self):
        for ind in range(0, self.LSTM.num_layers):
            weight = eval('self.LSTM.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('self.LSTM.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)

        if self.LSTM.bias:
            for ind in range(0, self.LSTM.num_layers):
                weight = eval('self.LSTM.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[self.LSTM.hidden_size: 2 * self.LSTM.hidden_size] = 1
                weight = eval('self.LSTM.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[self.LSTM.hidden_size: 2 * self.LSTM.hidden_size] = 1

        bias = np.sqrt(3.0 / self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -bias, bias)

    def init_hidden(self):
        r"""
        初始化隐藏层参数
        :param batch_size:  batch_size
        :return:
        """
        return (torch.randn(2, self.batch_size, self.hidden_size // 2),
                torch.randn(2, self.batch_size, self.hidden_size // 2))

    def _get_scores(self, sentences):
        '''
        得到序列的特征
        :param sentences: tensor [length, batch_size]
        :return: feats tensor [length, batch_size, labelset_size]
        '''

        self.hidden = self.init_hidden()
        # [length, batch_size] -> [length, batch_size, dim]
        sentences_embeddings = self.embeddings(sentences)
        sentences_embeddings = self.dropout1(sentences_embeddings)

        # outputs [length, batch_size, hidden_size]
        outputs, self.hidden = self.LSTM(sentences_embeddings, self.hidden)
        outputs = self.dropout2(outputs)

        # [length, batch_size, labelset_size]
        feats = self.output2label(outputs)
        return feats

    def _forward_all_logsumexp(self, scores, masks):
        r"""
        计算所有可能路径的log_sum_exp
        :param scores: tensor [length, batch_size, labelset_size]
            LSTM传过来的emit score
        :param masks: tensor [length, batch_size]
        :return: terminal_score: tensor [batch_size]
        """

        # 到当前单词，且状态为i的所有路径的log_sum_exp
        dp = torch.full((self.labelset_size, self.batch_size), -10000.)
        dp[self.label2idx[START_label]] = 0.

        for i in range(self.seq_length):
            # [batch_size, labelset_size]
            score = scores[i]
            # [batch_size] -> [batch_size, 1] -> [batch_size, labelset_size]
            mask = masks[i].unsqueeze(dim=1).expand(self.batch_size, self.labelset_size)
            # [labelset_size_from, batch_size, labelset_size_to]
            tmp = dp.transpose(0, 1).unsqueeze(dim=2).expand(self.labelset_size, self.batch_size, self.labelset_size) + \
                  score.unsqueeze(dim=0).expand(self.labelset_size, self.batch_size, self.labelset_size) + \
                  self.transitions.unsqueeze(dim=1).expand(self.labelset_size, self.batch_size, self.labelset_size)
            # [labelset_size_from, batch_size, labelset_size_to] -> [batch_size, labelset_size_to]
            tmp = log_sum_exp(tmp, dim=0)
            # mask为1的值更新，为0的不再更新
            dp.masked_scatter_(mask, tmp.masked_select(mask))

        # dp = dp + self.transitions[self.label2idx[STOP_label]]
        dp = log_sum_exp(dp, dim=1)

        return dp

    def _get_gold_score(self, scores: torch.tensor, labels, masks):
        '''
        计算出所提供的正确路径得分数
        :param scores: tensor [length, batch_size, labelset_size]
            LSTM传过来的emit score
        :param labels: tensor [length, batch_size]
            每个序列正确的路径, 已经加了start
        :param masks: tensor [length, batch_size]
        :return:
           scores: tensor [batch_size]
        '''

        dp = torch.zeros(self.batch_size)

        st = torch.full([1, self.batch_size], self.label2idx[START_label], dtype=torch.long)
        # [length + 1, batch_size]
        labels = torch.cat(
            [st, labels], dim=0
        )
        for i in range(self.seq_length):
            # [batch_size, labelset_size]
            score = scores[i]
            # [batch_size]
            mask = masks[i]
            tmp = dp + self.transitions[labels[i], labels[i + 1]] + score[
                range(self.batch_size), labels[i + 1]]
            # mask为1的值更新为新的tmp值，为0的不再更新
            dp.masked_scatter_(mask, tmp.masked_select(mask))

        # label最后一个永远是pad....
        # dp = dp + self.transitions[labels[-1], self.label2idx[STOP_label]]
        # print(time.time() - st)

        return dp

    def neg_log_likelihood(self, sentences, labels, masks):
        r"""
        计算正确路径的负对数似然概率

        :param sentences: tensor [length, batch_size]
        :param labels: tensor [length, batch_size]
            正确的label序列
        :param masks:tensor [length, batch_size]
        :return: FloatTensor
        """
        self.set_batch_seq_size(sentences)

        # [length, batch_size, labelset_size]
        feats = self._get_scores(sentences)

        forward_score = self._forward_all_logsumexp(feats, masks)
        gold_score = self._get_gold_score(feats, labels, masks)

        # print('forward_score: ', forward_score)
        # print('gold_score   :', gold_score)

        return (forward_score - gold_score).sum() / self.batch_size

    def _viterbi_decode(self, feats, masks):
        r'''
        使用维特比算法进行解码，找到最可能的序列结果

        :param feats: tensor [length, batch_size, labelset_size]
            LSTM传过来的特征向量
        :param masks: tensor [length, batch_size]
        :return: best_scores tensor [batch_size]
                best_paths tensor [length, batch_size]
        '''

        # 记录每个节点由哪个父节点转移过来
        parents = []
        # 到当前单词，且状态为i的所有路径中log_sum_exp最大的值
        dp = torch.full((self.labelset_size, self.batch_size), -10000.)
        # START_label has all of the score.
        dp[self.label2idx[START_label]] = 0.
        for i in range(feats.shape[1]):
            # [batch_size, labelset_size]
            feat = feats[i]
            # [batch_size] -> [batch_size, 1]
            mask = masks[i].unsqueeze(dim=1)
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
        # terminal_var = dp + self.transitions[:, self.label2idx[STOP_label]]
        terminal_var = dp

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

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.batch_size = tmp[1]
        self.seq_length = tmp[0]

    def predict(self, sentences, masks):
        r"""
        预测数据的最可能序列以及得分

        :param sentences: tensor [length, batch_size]
        :return:
            scores: tensor [batch_size]
            paths: list [tensor: [batch_size]....] with length=length
        """
        self.set_batch_seq_size(sentences)

        # Get the emission scores from the BiLSTM
        feats = self._get_scores(sentences)

        # Find the best path, given the features.
        scores, paths = self._viterbi_decode(feats, masks)
        return scores, paths


class BiLSTM_CRF_L(nn.Module):
    r"""
    Large LSTM，直接使用nn.Linear(hidden_dim, self.labelset_size * self.labelset_size)
        代替了转移矩阵，并且在制作数据集的时候采用label_i * labelset_size + label_(i + 1)
        的方法，可以一次计算出gold score，在之后也不用每次加trans，大大提高了运行速度，但是
        内存占用更大
    """

    def __init__(self, vocab_size, label2idx, embedding_dim, hidden_dim, num_layers, dropout_ratio):
        super(BiLSTM_CRF_L, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)
        self.num_layers = num_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.labelset_size = len(label2idx)
        self.label2idx = label2idx

        self.start_tag = label2idx[START_label]
        self.end_tag = label2idx[PAD_label]

        self.batch_size = 1
        self.seq_length = 1

        self.hidden2tag = nn.Linear(hidden_dim, self.labelset_size * self.labelset_size)

    def init_uniform(self):
        # LSTM
        r"""
        线性初始化网络
        :return:
        """
        for ind in range(0, self.LSTM.num_layers):
            weight = eval('self.LSTM.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('self.LSTM.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)

        if self.LSTM.bias:
            for ind in range(0, self.LSTM.num_layers):
                weight = eval('self.LSTM.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[self.LSTM.hidden_size: 2 * self.LSTM.hidden_size] = 1
                weight = eval('self.LSTM.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[self.LSTM.hidden_size: 2 * self.LSTM.hidden_size] = 1

        # embedding
        # nn.Embeddig.weight默认初始化方式就是N(0, 1)分布
        bias = np.sqrt(3.0 / self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -bias, bias)

        # Linear
        bias = np.sqrt(6.0 / (self.hidden2tag.weight.size(0) +
                              self.hidden2tag.weight.size(1)))
        nn.init.uniform_(self.hidden2tag.weight, -bias, bias)

    def rand_init_hidden(self):
        """
        随机初始化hidden
        """
        return torch.Tensor(
            torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2)), torch.Tensor(
            torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_seq_size(self, sentence):
        """
        :param sentence [length, batch_size]
        设置batch_size，seq_length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        加载预训练embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.embeddings.weight = nn.Parameter(pre_embeddings)

    def _get_gold_score(self, scores, targets, masks):
        r"""
        计算正确路径得分

        :param scores: [length, batch_size, labelset_size, labelset_size]
        :param targets: [length, batch_size]
        :param masks: [length, batch_size]
        :return: gold_score tensor
        """
        # [length, batch_size] -> [length, batch_size, 1]
        targets = targets.unsqueeze(dim=2)
        gold_score = torch.gather(scores.view(
            self.seq_length, self.batch_size, -1), 2, targets).view(self.seq_length,
                                                                    self.batch_size)  # seq_len * batch_size
        gold_score = gold_score.masked_select(masks).sum()
        return gold_score

    def _get_all_logsumexp(self, scores, masks):
        r"""
        计算所有路径的得分之和

        :param scores: [length, batch_size, labelset_size, labelset_size]
        :param masks: [length, batch_size]
        :return:
        """
        seq_iter = enumerate(scores)
        # [batch_size, labelset_size_from, labelset_size_to]
        _, inivalues = seq_iter.__next__()

        # [batch_size, labelset_size_to], ps: 不加clone会报错
        # 到当前单词，且状态为i的所有路径的log_sum_exp
        dp = inivalues[:, self.start_tag, :].clone()

        # 从正式的第一个label开始迭代
        for idx, cur_values in seq_iter:
            # [batch_size] -> [batch_size, labelset_size]
            mask = masks[idx].view(self.batch_size, 1).expand(self.batch_size, self.labelset_size)
            # cur_values: [batch_size, labelset_size_from, labelset_size_to]
            cur_values = cur_values + dp.contiguous().view(self.batch_size, self.labelset_size,
                                                           1).expand(self.batch_size, self.labelset_size,
                                                                     self.labelset_size)
            # [batch_size, from_target, to_target] -> [batch_size, to_target]
            tmp = log_sum_exp(cur_values, dim=1)

            # 0保留自身值，1采用新的source值
            dp.masked_scatter_(mask, tmp.masked_select(mask))

        dp = dp[:, self.end_tag].sum()

        return dp

    def neg_log_likelihood(self, sentences, targets, masks, hidden=None):
        r"""
        计算损失函数
        :param sentences: [length, batch_size]
        :param targets: [length, batch_size]
        :param masks: [length, batch_size]
        :param hidden:
        :return:
        """
        # [length, batch_size, labelset_size, labelset_size]
        crf_scores = self.forward(sentences)
        gold_score = self._get_gold_score(crf_scores, targets, masks)

        forward_score = self._get_all_logsumexp(crf_scores, masks)

        loss = (forward_score - gold_score) / self.batch_size
        # print(loss)

        return loss

    def _viterbi_decode(self, crf_scores, masks):
        r'''
        使用维特比算法进行解码，找到最可能的序列结果

        :param crf_scores: tensor [length, batch_size, labelset_size, labelset_size]
            LSTM传过来的emit score + trans score
        :param masks: tensor [length, batch_size]
        :return: scores tensor [batch_size]
                 paths [batch_size, seq_length - 1]
        '''

        # 方便后面的mask fill
        masks = ~masks
        path = torch.LongTensor(self.seq_length - 1, self.batch_size)

        seq_iter = enumerate(crf_scores)
        # [batch_size, from_labelset_size, to_labelset_size]
        _, inivalues = seq_iter.__next__()
        # 只保留start的初始得分, [batch_size, to_labelset_size]
        forscores = inivalues[:, self.start_tag, :].clone()

        parents = []
        # 从正式的第一个label开始迭代
        for idx, cur_values in seq_iter:
            # [batch_size] -> [batch_size, labelset_size]
            mask = masks[idx].view(self.batch_size, 1).expand(self.batch_size, self.labelset_size)
            # cur_values: [batch_size, from_target, to_target]
            cur_values = cur_values + forscores.contiguous().view(self.batch_size, self.labelset_size,
                                                                  1).expand(self.batch_size, self.labelset_size,
                                                                            self.labelset_size)
            forscores, cur_parent = torch.max(cur_values, 1)
            # [batch_size, to_target], mask是1是直接pad
            cur_parent.masked_fill_(mask, self.end_tag)
            parents.append(cur_parent)

        pointer = parents[-1][:, self.end_tag]
        path[-1] = pointer
        for idx in range(len(parents) - 2, -1, -1):
            back_point = parents[idx]
            index = pointer.contiguous().view(-1, 1)
            pointer = torch.gather(back_point, 1, index).view(-1)
            path[idx] = pointer

        return forscores, path.transpose(0, 1)

    def predict(self, sentences, masks, hidden=None):
        r"""
        进行预测，计算得分和最优路径

        :param sentences: [length, batch_size]
        :param masks: [length, batch_size]
        :return:
        """
        self.eval()

        crf_scores = self.forward(sentences)
        scores, path = self._viterbi_decode(crf_scores, masks)

        return scores, path

    def forward(self, sentences, hidden=None):
        r"""
        计算crf_scores

        :param sentences: [length, batch_size]
        :param hidden: LSTM的初始隐藏层
        :return: crf_scores [length, batch_size, labelset_size_from, labelset_size_to]
            crf_scores[0, 0, 1, 10]: 第一个句的第一个单词 从label_1 -> label_10的emit_score + trans_score
        """
        self.set_batch_seq_size(sentences)
        embeds = self.embeddings(sentences)
        d_embeds = self.dropout1(embeds)
        # [length, batch_size, hidden_size]
        lstm_out, hidden = self.LSTM(d_embeds, hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_scores = self.hidden2tag(d_lstm_out).view(-1, self.labelset_size, self.labelset_size)
        crf_scores = crf_scores.view(self.seq_length, self.batch_size, self.labelset_size, self.labelset_size)

        return crf_scores
