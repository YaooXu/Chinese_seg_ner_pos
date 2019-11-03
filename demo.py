import time
import os
import torch.optim as optim
from utils import generate_corpus, construct_bucket_mean, read_corpus, adjust_learning_rate, write_gold_file
import torch
import tqdm
import itertools
from BiLSTM_CRF import BiLSTM_CRF_S
from config import parse_args
import torch.nn as nn
from evaluate import evaluate, evaluate_with_perl, predict_write


def complete_dict(label2idx, add_label2idx):
    for label in add_label2idx:
        if label not in label2idx:
            label2idx[label] = len(label2idx)


def run_demo(sentence):
    test_features = [list(sentence)]
    # dummy label 没用
    test_labels = [['<start>' for _ in range(len(test_features[0]))]]

    test_datasets = construct_bucket_mean(
        test_features, test_labels, feature2idx, label2idx)[0]
    test_dataset_loader = [torch.utils.data.DataLoader(
        test_datasets, args.batch_size, shuffle=False, drop_last=False)]

    for _, batch in enumerate(itertools.chain.from_iterable(test_dataset_loader)):
        batch = tuple(t.to(device) for t in batch)
        features, labels, masks = batch
        features_v, labels_v, masks_v = features.transpose(0, 1), labels.transpose(0, 1), masks.transpose(0, 1)
        scores, predict_labels = model.predict(features_v, masks_v)
        for j in range(labels.shape[0]):
            feature, predict_label, mask = features[j], predict_labels[j], masks[j]
            line = ''
            pos = ''
            for k in range(features.shape[1]):
                # 多一个<eof>
                if k + 1 == features.shape[1] or mask[k + 1].item() == 0:
                    print(line)
                    print(pos)
                    break
                else:
                    if idx2label[predict_label[k].item()][0] in ('B', 'S'):
                        pos += idx2label[predict_label[k].item()][2:] + ' '
                        if k != 0:
                            line += ' ' + idx2feature[feature[k].item()]
                        else:
                            line += idx2feature[feature[k].item()]
                    else:
                        line += idx2feature[feature[k].item()]


if __name__ == '__main__':
    args = parse_args()

    #
    # if args.gpu >= 0 and torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     torch.cuda.set_device(args.gpu)
    #     # 设置为默认在GPU创建
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    device = torch.device('cpu')

    checkpoint_file = None
    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']

            feature2idx = checkpoint_file['feature2idx']
            label2idx = checkpoint_file['label2idx']

            # 读取模型参数
            try:
                args.embedding_dim = checkpoint_file['embedding_dim']
                args.hidden_size = checkpoint_file['hidden_size']
                args.layers = checkpoint_file['layers']
                args.drop_out = checkpoint_file['drop_out']
                print('args of model load successfully')
            except:
                print('fail to args of model load')

        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
            exit(-1)
    else:
        # 制作语料集
        print("no checkpoint")
        exit(-1)

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}
    # 加载模型
    model = BiLSTM_CRF_S(len(feature2idx), label2idx, args.embedding_dim, args.hidden_size, args.layers, args.drop_out)
    model.to(device)

    # 加载模型参数
    if checkpoint_file is not None:
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        exit(-1)

    model.eval()

    run_demo('中共中央')
