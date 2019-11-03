import tqdm
import itertools
import torch
import torch.utils.data
import os
from dataset import *

STOP_label = '<stop>'
START_label = '<start>'
PAD_label = '<pad>'
UNKOWN_label = '<unk>'
EOF_label = '<eof>'


def log_sum_exp(vec: torch.tensor, dim=0):
    r"""
    计算向量某个维度上的logsumexp
    :param vec: Tensor
    :param dim: 在哪个维度计算log_sum_exp, 默认为0
    :return: Tensor (1, vec.shape[1])
    """

    max_score = vec.max(dim=dim, keepdim=True)[0]
    return max_score.squeeze(dim=dim) + torch.log(torch.sum(torch.exp(vec - max_score), dim=dim))


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])

        elif len(tmp_fl) > 0:
            if len(tmp_fl) >= 2:
                features.append(tuple(tmp_fl))
                labels.append(tuple(tmp_ll))
            tmp_fl = list()
            tmp_ll = list()

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels


def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold
    """
    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(
        feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (
            ind + 1) for ind in range(0, len(shrinked_feature_count))}

    # inserting unk to be 0 encoded
    feature_map[UNKOWN_label] = 0
    # inserting eof
    feature_map[EOF_label] = len(feature_map)
    feature_map[START_label] = len(feature_map)
    return feature_map


def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        train_lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary

    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    feature_map = dict()
    label_map = dict()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] not in feature_map:
                feature_map[line[0]] = len(feature_map) + 1  # 0 is for unk

            tmp_ll.append(line[-1])
            if line[-1] not in label_map:
                label_map[line[-1]] = len(label_map)

        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    label_map[START_label] = len(label_map)
    label_map[PAD_label] = len(label_map)

    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        # inserting unk to be 0 encoded
        feature_map[UNKOWN_label] = 0
        # inserting eof
        feature_map[EOF_label] = len(feature_map)
        feature_map[START_label] = len(feature_map)

    return features, labels, feature_map, label_map


def encode_safe(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation with unk
    """
    lines = list(
        map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines


def encode_lines(input_lines, word_dict):
    """
    encode list of strings into word-level representation
    """
    lines = list(
        map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines


def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """
    lines_len = list(map(lambda t: len(t) + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    try:
        # 测试集上数据太少时
        lower_line = list(filter(lambda t: t < average, lines_len))
        upper_line = list(filter(lambda t: t >= average, lines_len))
        lower_average = int(sum(lower_line) / len(lower_line))
        upper_average = int(sum(upper_line) / len(upper_line))
        max_len = max(lines_len)
    except:
        lower_average, upper_average, max_len = average, average, average

    print([lower_average, average, upper_average, max_len])

    return [lower_average, average, upper_average, max_len]


def construct_bucket_mean(input_features, input_label, feature2idx, label2idx, Large=False):
    """
    Large: 默认为False, 即为BILSTM_CRF_L制作数据集
    """
    label_size = len(label2idx)

    # encode and padding
    features = encode_safe(input_features, feature2idx, feature2idx[UNKOWN_label])

    # # TODO:DEBUG
    # features = list(map(lambda t: [feature2idx[START_label]] + list(t), features))

    labels = encode_lines(input_label, label2idx)
    if Large:
        # labels = list(map(lambda t: [label2idx[START_label]] + list(t), labels))
        labels = list(map(lambda t: [label2idx[START_label]] + list(t) + [label2idx[PAD_label]], labels))
    thresholds = calc_threshold_mean(features)

    pad_feature = feature2idx[EOF_label]
    pad_label = label2idx[PAD_label]

    buckets = [[[], [], []] for _ in range(len(thresholds))]
    for feature, label in zip(features, labels):
        cur_len = len(feature)
        idx = 0
        cur_len_1 = cur_len + 1
        # 根据当前句子的长度选取不同的阈值
        while thresholds[idx] < cur_len_1:
            idx += 1
        if Large:
            buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
            buckets[idx][1].append([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
                label[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
                                           thresholds[idx] - cur_len_1))
            buckets[idx][2].append([1] * cur_len_1 + [0] *
                                   (thresholds[idx] - cur_len_1))
        else:
            buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))

            buckets[idx][1].append(label + [pad_label] * (thresholds[idx] - cur_len))

            buckets[idx][2].append([1] * cur_len + [0] *
                                   (thresholds[idx] - cur_len))

    # 注意mask只能是BoolTensor
    bucket_dataset = [
        CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.tensor(bucket[2], dtype=torch.bool))
        for bucket in buckets]

    return bucket_dataset


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def write_gold_file(dataloader: torch.utils.data.Dataset, train_file, idx2label, idx2feature, tgt_file, rewrite=0):
    r"""
        把按长度分块之后的数据重新写一份金标文件,如果当前文件已经存在则重新生成

    :param dataloader:
    :param train_file:
    :param idx2label:
    :param idx2feature:
    :param tgt_file:
    :param rewrite: 1为覆盖重写，0为若已经存在则跳过
    :return:
    """
    parent_dir = os.path.split(train_file)[0]
    tgt_file = os.path.join(parent_dir, tgt_file)

    if rewrite == 0 and os.path.exists(tgt_file):
        print('%s already exits' % tgt_file)
        return tgt_file

    print('Writing %s ...' % tgt_file)
    with open(tgt_file, 'w', encoding='utf8') as f:
        for i, batch in tqdm.tqdm(enumerate(
                itertools.chain.from_iterable(dataloader))):
            features, labels, masks = batch
            for j in range(features.shape[0]):
                feature, label, mask = features[j], labels[j], masks[j]
                line = ''
                for k in range(features.shape[1]):
                    # 多一个<eof>
                    if k == features.shape[1] or mask[k].item() == 0:
                        f.write(line + '\n')
                        break
                    else:
                        if idx2label[label[k].item()][0] in ('B', 'S') and k != 0:
                            line += ' ' + idx2feature[feature[k].item()]
                        else:
                            line += idx2feature[feature[k].item()]
    return tgt_file
