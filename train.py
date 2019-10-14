import time
import os
import torch.optim as optim
from utils import generate_corpus, construct_bucket_mean, read_corpus, adjust_learning_rate
import torch
import tqdm
import itertools
from BiLSTM_CRF import BiLSTM_CRF
from config import parse_args
import torch.nn as nn
from evaluate import evaluate

if __name__ == '__main__':
    args = parse_args()

    model_name = '{:}_{:}_{:}_{:.2f}_{:}'.format(args.batch_size, args.hidden_size, args.embedding_dim, args.drop_out,
                                                 args.layers)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu)
        # 设置为默认在GPU创建
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    print('setting:')
    print(device)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()

    checkpoint_file = None
    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            feature2idx = checkpoint_file['feature2idx']
            label2idx = checkpoint_file['label2idx']
            train_features, train_labels = read_corpus(train_lines)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
    else:
        # build corpus
        train_features, train_labels, feature2idx, label2idx = generate_corpus(train_lines)

    # 制作验证集合测试集
    num_train = int(len(train_features) * (1.0 - args.dev_ratio))
    train_features, dev_features = train_features[:num_train], train_features[num_train:]
    train_labels, dev_labels = train_labels[:num_train], train_labels[num_train:]
    test_features, test_labels, test_feature2idx, test_label2idx = generate_corpus(test_lines)

    # 部分test中的标签train中没有
    for label in test_label2idx:
        if label not in label2idx:
            label2idx[label] = len(label2idx)

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}

    dataset = construct_bucket_mean(
        train_features, train_labels, feature2idx, label2idx)
    dev_dataset = construct_bucket_mean(
        dev_features, dev_labels, feature2idx, label2idx)
    test_dataset = construct_bucket_mean(
        test_features, test_labels, feature2idx, label2idx)

    train_dataset_loader = [torch.utils.data.DataLoader(
        tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(
        tup, args.batch_size, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(
        tup, args.batch_size, shuffle=False, drop_last=False) for tup in test_dataset]

    model = BiLSTM_CRF(len(feature2idx), label2idx, args.embedding_dim, args.hidden_size, args.layers, args.drop_out)
    model.to(device)

    # evaluate(model, test_dataset_loader, idx2feature, idx2label, device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    if checkpoint_file is not None:
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        # TODO : 加载embedding以及选择初始化方式
        pass

    if args.update == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if checkpoint_file is not None and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    tot_length = sum(map(lambda t: len(t), train_dataset_loader))
    best_F1 = float('-inf')
    best_acc = float('-inf')
    track_list = []
    start_time = time.time()
    end_epoch = args.start_epoch + args.epoch
    epoch_list = range(args.start_epoch, end_epoch)
    patience_count = 0

    for epoch_idx, args.start_epoch in enumerate(epoch_list):
        epoch_loss = 0
        model.train()

        for i, batch in enumerate(
                itertools.chain.from_iterable(train_dataset_loader)):
            model.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            features, labels, masks = batch
            batch_size = features.shape[0]

            loss = model.neg_log_likelihood(features, labels, masks)
            epoch_loss += loss

            print('loss: {:.4f}'.format(loss.item()), ' {:} / {:} iteration '.format(i + 1, tot_length),
                  ' {:} / {:} epoch'.format(epoch_idx, end_epoch))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # 调整学习率
        adjust_learning_rate(
            optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        epoch_loss /= batch_size
        print('epoch average loss %f' % epoch_loss)

        print('train result')
        evaluate(model, train_dataset_loader, idx2feature, idx2label, device)

        print('dev result')
        dev_P, dev_R, dev_F1, dev_ER = evaluate(model, dev_dataset_loader, idx2feature, idx2label, device)

        track_list.append({
            'loss': epoch_loss,
            'F1': dev_F1,
            'accuracy': dev_P,
            'error': dev_ER
        })

        if dev_F1 > best_F1:
            # 保存模型
            patience_count = 0
            print('保存更好的模型...')
            best_F1 = dev_F1
            save_file_path = os.path.join(args.checkpoint_dir,
                                          'save_{}_{}.pth'.format(epoch_idx, i + 1))
            states = {
                'epoch': args.start_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'track_list': track_list,
                'feature2idx': feature2idx,
                'label2idx': label2idx
            }
            torch.save(states, save_file_path)
        else:
            patience_count += 1

    print('Final result : Acc : %{:.4f} , F1 : %{:.4f}'.format(best_acc, best_F1))
