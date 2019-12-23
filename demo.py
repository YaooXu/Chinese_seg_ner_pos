import time
import os
import torch.optim as optim
from utils import generate_corpus, construct_bucket_for_train, read_corpus, adjust_learning_rate, write_gold_file
import torch
import tqdm
import itertools
from BiLSTM_CRF import BiLSTM_CRF_L
from config import parse_args
import torch.nn as nn
from evaluate import evaluate, evaluate_with_perl, predict_write
import argparse


def complete_dict(label2idx, add_label2idx):
    for label in add_label2idx:
        if label not in label2idx:
            label2idx[label] = len(label2idx)


def run_demo(sentence, origin_texts=None):
    test_features = [list(sentence)]
    # dummy label 没用
    test_labels = [['<start>' for _ in range(len(test_features[0]))]]

    test_datasets = construct_bucket_for_train(
        test_features, test_labels, feature2idx, label2idx)[0]
    test_dataset_loader = [torch.utils.data.DataLoader(
        test_datasets, 1, shuffle=False, drop_last=False)]

    for idx, batch in enumerate(itertools.chain.from_iterable(test_dataset_loader)):
        batch = tuple(t.to(device) for t in batch)
        features, labels, masks = batch
        features_v, labels_v, masks_v = features.transpose(0, 1), labels.transpose(0, 1), masks.transpose(0, 1)
        scores, predict_labels = model.predict(features_v, masks_v)
        batch_size = labels.shape[0]
        # 原始文本内容，避免最终结果出现<unk>
        if origin_texts:
            origin_text = origin_texts[idx * batch_size:(idx + 1) * batch_size]

        for j in range(batch_size):
            if origin_texts:
                origin_line = origin_text[j]

            feature, predict_label, mask = features[j], predict_labels[j], masks[j]
            line = ''
            label = []
            length = feature.shape[0]
            for k in range(length):
                if k + 1 == length or mask[k + 1].item() == 0:
                    break
                else:
                    if origin_texts:
                        content = origin_line[k]
                    else:
                        content = idx2feature[feature[k].item()]

                    if idx2label[predict_label[k].item()][0] in ('B', 'S') and k != 0:
                        line += ' ' + content
                        label.append(idx2label[predict_label[k].item()][2:])
                    else:
                        line += content
            print(line)
            print(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing with BLSTM-CRF')
    parser.add_argument(
        '--test_file', default='./icwb2-data/testing/pku_test.utf8', help='path to test file')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--load_check_point', default='./checkpoint/128_300_300_0.30_1_sgd_0.200/best.pth',
                        help='path of checkpoint')
    parser.add_argument('--large', type=int, default=1,
                        help='whether to use large model')
    parser.add_argument('--target_file', type=str, default='./result',
                        help='the predicted result file')
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu)
        # 设置为默认在GPU创建
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
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
                exit(-1)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
            exit(-1)
    else:
        print("no checkpoint")
        exit(-1)

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}
    # 加载模型
    model = BiLSTM_CRF_L(len(feature2idx), label2idx, args.embedding_dim, args.hidden_size, args.layers, args.drop_out)
    model.to(device)

    # 加载模型参数
    if checkpoint_file is not None:
        print('Loding model state_dict...')
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        exit(-1)
    # 加载模型参数
    if checkpoint_file is not None:
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        exit(-1)

    model.eval()

    run_demo('分词是自然语言处理的重要部分')
    run_demo('国家主席江泽民近日成功抵达美国进行国事访问。')
    run_demo('中国表示强烈谴责美国对我国内政的干扰')
    run_demo('近日美国与中国就台海事件进行交涉')