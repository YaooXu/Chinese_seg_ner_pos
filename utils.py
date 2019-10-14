import tqdm
import itertools
import torch
from BiLSTM_CRF import CRFDataset


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = []
    labels = []
    tmp_fl = []
    tmp_ll = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = []
            tmp_ll = []

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
    feature_map['<unk>'] = 0
    # inserting eof
    feature_map['<eof>'] = len(feature_map)
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

    label_map['<start>'] = len(label_map)
    label_map['<pad>'] = len(label_map)
    label_map['<stop>'] = len(label_map)

    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        # inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        # inserting eof
        feature_map['<eof>'] = len(feature_map)

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
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t >= average, lines_len))
    lower_average = int(sum(lower_line) / len(lower_line))
    upper_average = int(sum(upper_line) / len(upper_line))
    max_len = max(lines_len)
    return [lower_average, average, upper_average, max_len]


def construct_bucket(input_features, input_labels, thresholds, pad_feature, pad_label, label_size):
    """
    Construct bucket by thresholds for viterbi decode, word-level only
    """
    buckets = [[[], [], []] for _ in range(len(thresholds))]
    for feature, label in zip(input_features, input_labels):
        cur_len = len(feature)
        idx = 0
        cur_len_1 = cur_len + 1
        # 根据当前句子的长度选取不同的阈值
        while thresholds[idx] < cur_len_1:
            idx += 1

        buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))

        # 很巧妙的设计，但过于奇技淫巧
        # 在算gold score时可以直接一步算出
        # buckets[idx][1].append([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
        #     label[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
        #                                thresholds[idx] - cur_len_1))
        buckets[idx][1].append(label + [pad_label] * (thresholds[idx] - cur_len))

        # 注意mask是byte tensor
        buckets[idx][2].append([1] * cur_len_1 + [0] *
                               (thresholds[idx] - cur_len_1))

    bucket_dataset = [CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.tensor(bucket[2], dtype=torch.bool))
                      for bucket in buckets]
    return bucket_dataset


def construct_bucket_mean(input_features, input_label, feature2idx, label2idx):
    """
    Construct bucket by mean for viterbi decode, word-level only
    """
    # encode and padding

    features = encode_safe(input_features, feature2idx, feature2idx['<unk>'])
    labels = encode_lines(input_label, label2idx)

    # 并没有标记stop?
    # 在网络中自行添加
    # labels = list(map(lambda t: [label2idx['<start>']] + list(t), labels))

    thresholds = calc_threshold_mean(features)

    return construct_bucket(features, labels, thresholds, feature2idx['<eof>'], label2idx['<pad>'], len(label2idx))


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


