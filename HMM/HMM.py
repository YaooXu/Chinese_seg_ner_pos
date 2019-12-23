from functools import reduce
import random
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
from utils import encode_lines
from evaluate import evaluate_with_perl


def preprocess_text(file):
    with open(file, encoding='utf8') as f:
        lines = f.readlines()

    features = []
    lengths = []
    label_set = set()
    tmp_fl = list()
    tmp_ll = list()
    feature2idx = {}

    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] not in feature2idx:
                feature2idx[line[0]] = len(feature2idx)

            tmp_ll.append(line[-1])
            label_set.add(line[-1])

        elif len(tmp_fl) > 0:
            if len(tmp_fl) >= 2:
                features.append(tuple(tmp_fl))
                lengths.append(len(tmp_fl))
            tmp_fl = list()
            tmp_ll = list()

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        lengths.append(len(tmp_fl))
    return features, feature2idx, lengths, label_set


def train(train_features, feature2idx):
    train_features = encode_lines(train_features, feature2idx)
    print('Begin to learn...')
    model = hmm.GaussianHMM(3, n_iter=100000, tol=0.001, verbose=True)
    X = np.concatenate(train_features).reshape(-1, 1)
    model.fit(X, train_lengths)

    joblib.dump(model, "filename.pkl")


def test(dev_features, dev_lengths):
    model = joblib.load("filename.pkl")
    res = model.decode(np.concatenate(dev_features).reshape(-1, 1), dev_lengths)
    res = res[1]
    dev_lengths.insert(0, 0)

    labels = []
    cur = 0
    for i in range(len(dev_lengths) - 1):
        labels.append(res[cur:cur + dev_lengths[i + 1]])
        cur += dev_lengths[i + 1]

    with open('tmp', 'w') as f:
        for label, dev_feature in zip(labels, dev_features):
            line = ''
            for i in range(len(label)):
                if idx2label[label[i]][0] in ('B', 'S') and i != 0:
                    line += ' ' + idx2feature[dev_feature[i]]
                else:
                    line += idx2feature[dev_feature[i]]
            f.write(line + '\n')

    evaluate_with_perl('./dev_order_gold', './tmp', 'log', 0)


if __name__ == '__main__':
    train_file = r'/home/yxu/Seg_ner_pos/data/little_renmin/train'
    dev_file = r'/home/yxu/Seg_ner_pos/data/little_renmin/dev'
    train_features, train_feature2idx, train_lengths, train_label_set = preprocess_text(train_file)
    dev_features, test_feature2idx, dev_lengths, dev_label_set = preprocess_text(dev_file)

    label_set = train_label_set | dev_label_set
    label2idx = {}
    for i in label_set:
        label2idx[i] = len(label2idx)

    feature2idx = train_feature2idx
    for i in test_feature2idx:
        if i not in feature2idx:
            feature2idx[i] = len(feature2idx)

    label2idx = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}

    # train_features, train_lengths = train_features[:10], train_lengths[:10]

    train_features = encode_lines(train_features, feature2idx)
    dev_features = encode_lines(dev_features, feature2idx)

    test(dev_features, dev_lengths)