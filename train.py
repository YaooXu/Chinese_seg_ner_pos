import time
import os
import torch.optim as optim
from utils import generate_corpus, construct_bucket_for_train, read_corpus, adjust_learning_rate, write_gold_file
import torch
import tqdm
import itertools
from BiLSTM_CRF import BiLSTM_CRF_S, BiLSTM_CRF_L

from config import parse_args
import torch.nn as nn
from evaluate import evaluate, evaluate_with_perl, predict_write, evaluate_by_file


def complete_dict(label2idx, add_label2idx):
    for label in add_label2idx:
        if label not in label2idx:
            label2idx[label] = len(label2idx)


if __name__ == '__main__':
    args = parse_args()

    model_name = '{:}_{:}_{:}_{:.2f}_{:}_{:}_{:.3f}'.format(args.batch_size, args.hidden_size, args.embedding_dim, args.drop_out,
                                                 args.layers, args.update, args.lr)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    args.log_path = os.path.join(args.checkpoint_dir, 'log')
    # 不然最后面会带一个\n!
    args.load_check_point = args.load_check_point.strip()

    # 创建checkpoint目录
    if not os.path.exists(args.checkpoint_dir):
        print('Making dir %s' % args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu)
        # 设置为默认在GPU创建
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    print('Setting:')
    print(device)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.dev_file, 'r', encoding='utf-8') as f:
        dev_lines = f.readlines()
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()

    checkpoint_file = None
    if args.load_check_point:
        print(args.load_check_point)
        print(os.path.exists(args.load_check_point))

        if os.path.exists(args.load_check_point):
            print("Loading checkpoint: '{}'".format(args.load_check_point))
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
                print('Args of model load successfully')
            except:
                print('Fail to load args of model')

            train_features, train_labels = read_corpus(train_lines)
            test_features, test_labels = read_corpus(test_lines)
            dev_features, dev_labels = read_corpus(dev_lines)
        else:
            print('No checkpoint found at: {}'.format(args.load_check_point))
            exit(-1)
    else:
        # 制作语料集
        print('Building corpus...')
        train_features, train_labels, feature2idx, label2idx = generate_corpus(train_lines)
        test_features, test_labels, test_feature2idx, test_label2idx = generate_corpus(test_lines)
        dev_features, dev_labels, dev_feature2idx, dev_label2idx = generate_corpus(dev_lines)

        # 部分test中的标签train中没有
        complete_dict(label2idx, test_label2idx)
        complete_dict(label2idx, dev_label2idx)
        complete_dict(feature2idx, test_feature2idx)
        complete_dict(feature2idx, dev_feature2idx)

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}

    # 制作数据集
    # [CRFdataset * 4]
    train_datasets = construct_bucket_for_train(
        train_features, train_labels, feature2idx, label2idx, Large=args.large)
    dev_datasets = construct_bucket_for_train(
        dev_features, dev_labels, feature2idx, label2idx, Large=args.large)
    test_datasets = construct_bucket_for_train(
        test_features, test_labels, feature2idx, label2idx, Large=args.large)

    train_dataset_loader = [torch.utils.data.DataLoader(
        tup, args.batch_size, shuffle=False, drop_last=False) for tup in train_datasets]
    dev_dataset_loader = [torch.utils.data.DataLoader(
        tup, 64, shuffle=False, drop_last=False) for tup in dev_datasets]
    test_dataset_loader = [torch.utils.data.DataLoader(
        tup, 64, shuffle=False, drop_last=False) for tup in test_datasets]

    # 重写金标文件
    REWRITE = args.rewrite
    print('Writing gold file...')
    train_gold = write_gold_file(train_dataset_loader, args.train_file, idx2label, idx2feature, 'train_gold', REWRITE)
    dev_gold = write_gold_file(dev_dataset_loader, args.train_file, idx2label, idx2feature, 'dev_gold', REWRITE)
    test_gold = write_gold_file(test_dataset_loader, args.train_file, idx2label, idx2feature, 'test_gold', REWRITE)

    # 加载模型
    print('Loading model...')
    model = BiLSTM_CRF_L(len(feature2idx), label2idx, args.embedding_dim, args.hidden_size, args.layers, args.drop_out)
    model.to(device)

    # 加载模型参数
    if checkpoint_file is not None:
        print('Load state dict...')
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        print('Rand initing...')
        model.init_uniform()

    # 加载优化器以及参数
    if args.update == 'sgd':
        print('Loading Sgd optimizer...')
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        print('Loading Adam optimizer...')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if checkpoint_file is not None and args.load_opt:
        args.load_opt = args.load_opt.strip()
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    tot_iters = sum(map(lambda t: len(t), train_dataset_loader))
    try:
        # 部分早期模型没有这两个属性
        best_F1 = checkpoint_file['best_F1']
        best_acc = checkpoint_file['best_acc']
        print('Best F1: %f best_acc: %f' % (best_F1, best_acc))
    except:
        best_F1 = float('-inf')
        best_acc = float('-inf')
    start_time = time.time()
    end_epoch = args.epoch
    epoch_list = range(args.start_epoch + 1, end_epoch)
    patience_count = 0

    # 在dev上的最原始效果
    print('Origin performance')
    tmp_file = predict_write(model, dev_dataset_loader, idx2feature, idx2label, device)
    dev_R, dev_P, dev_F1 = evaluate_with_perl(dev_gold, tmp_file, args.log_path, 0)

    for epoch_idx, args.start_epoch in enumerate(epoch_list):
        epoch_start_time = time.time()
        epoch_loss = 0
        loss = 0
        model.train()

        for i, batch in enumerate(
                itertools.chain.from_iterable(train_dataset_loader)):
            model.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            features, labels, masks = batch
            batch_size = features.shape[0]
            features_v, labels_v, masks_v = features.transpose(0, 1), labels.transpose(0, 1), masks.transpose(0, 1)
            loss = model.neg_log_likelihood(features_v, labels_v, masks_v)

            print('loss: {:.4f}'.format(loss.item()), ' {:} / {:} iteration '.format(i + 1, tot_iters),
                  ' {:} / {:} epoch'.format(args.start_epoch, end_epoch))

            loss.backward()
            # 不梯度剪裁很容易出现梯度爆炸梯度消失！
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            epoch_loss += loss.item()

        # 调整学习率
        adjust_learning_rate(
            optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        epoch_loss /= tot_iters
        print('Epoch average loss %f, takes %f s' % (epoch_loss, time.time() - epoch_start_time))

        print('Dev result')
        # 这个效率太低了...
        # dev_P, dev_R, dev_F1, dev_ER = evaluate(model, dev_dataset_loader, idx2feature, idx2label, device,
        #                                         args.log_path)
        tmp_file = predict_write(model, dev_dataset_loader, idx2feature, idx2label, device)
        dev_R, dev_P, dev_F1 = evaluate_with_perl(dev_gold, tmp_file, args.log_path, args.start_epoch, loss=epoch_loss, dev=True)

        # tmp_file = predict_write(model, train_dataset_loader, idx2feature, idx2label, device)
        # train_R, train_P, train_F1 = evaluate_with_perl(train_gold, tmp_file, args.log_path, args.start_epoch, dev=False)

        states = {
            'epoch': args.start_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'feature2idx': feature2idx,
            'label2idx': label2idx,
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'layers': args.layers,
            'drop_out': args.drop_out,
            'best_F1': best_F1,
            'best_acc': best_acc
        }

        save_file_path = os.path.join(args.checkpoint_dir, 'current.pth')
        print('Saving model to ', save_file_path)
        torch.save(states, save_file_path)

        if dev_F1 > best_F1:
            # 保存模型
            patience_count = 0

            best_F1 = dev_F1
            best_acc = best_F1

            save_file_path = os.path.join(args.checkpoint_dir, 'best.pth')
            print('Saving model to ', save_file_path)

            torch.save(states, save_file_path)
        else:
            patience_count += 1
            print(patience_count)
            if patience_count == args.patience:
                print('Stop training early')
                break

    print('Final result : Acc : %{:.4f} , F1 : %{:.4f}'.format(best_acc, best_F1))
import time
import os
import torch.optim as optim
from utils import generate_corpus, construct_bucket_for_train, read_corpus, adjust_learning_rate, write_gold_file
import torch
import tqdm
import itertools
from BiLSTM_CRF import BiLSTM_CRF_S, BiLSTM_CRF_L

from config import parse_args
import torch.nn as nn
from evaluate import evaluate, evaluate_with_perl, predict_write, evaluate_by_file


def complete_dict(label2idx, add_label2idx):
    for label in add_label2idx:
        if label not in label2idx:
            label2idx[label] = len(label2idx)


if __name__ == '__main__':
    args = parse_args()

    model_name = '{:}_{:}_{:}_{:.2f}_{:}_{:}_{:.3f}'.format(args.batch_size, args.hidden_size, args.embedding_dim, args.drop_out,
                                                 args.layers, args.update, args.lr)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    args.log_path = os.path.join(args.checkpoint_dir, 'log')
    # 不然最后面会带一个\n!
    args.load_check_point = args.load_check_point.strip()

    # 创建checkpoint目录
    if not os.path.exists(args.checkpoint_dir):
        print('Making dir %s' % args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu)
        # 设置为默认在GPU创建
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    print('Setting:')
    print(device)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.dev_file, 'r', encoding='utf-8') as f:
        dev_lines = f.readlines()
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()

    checkpoint_file = None
    if args.load_check_point:
        print(args.load_check_point)
        print(os.path.exists(args.load_check_point))

        if os.path.exists(args.load_check_point):
            print("Loading checkpoint: '{}'".format(args.load_check_point))
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
                print('Args of model load successfully')
            except:
                print('Fail to load args of model')

            train_features, train_labels = read_corpus(train_lines)
            test_features, test_labels = read_corpus(test_lines)
            dev_features, dev_labels = read_corpus(dev_lines)
        else:
            print('No checkpoint found at: {}'.format(args.load_check_point))
            exit(-1)
    else:
        # 制作语料集
        print('Building corpus...')
        train_features, train_labels, feature2idx, label2idx = generate_corpus(train_lines)
        test_features, test_labels, test_feature2idx, test_label2idx = generate_corpus(test_lines)
        dev_features, dev_labels, dev_feature2idx, dev_label2idx = generate_corpus(dev_lines)

        # 部分test中的标签train中没有
        complete_dict(label2idx, test_label2idx)
        complete_dict(label2idx, dev_label2idx)
        complete_dict(feature2idx, test_feature2idx)
        complete_dict(feature2idx, dev_feature2idx)

    idx2label = {item[1]: item[0] for item in label2idx.items()}
    idx2feature = {item[1]: item[0] for item in feature2idx.items()}

    # 制作数据集
    # [CRFdataset * 4]
    train_datasets = construct_bucket_for_train(
        train_features, train_labels, feature2idx, label2idx, Large=args.large)
    dev_datasets = construct_bucket_for_train(
        dev_features, dev_labels, feature2idx, label2idx, Large=args.large)
    test_datasets = construct_bucket_for_train(
        test_features, test_labels, feature2idx, label2idx, Large=args.large)

    train_dataset_loader = [torch.utils.data.DataLoader(
        tup, args.batch_size, shuffle=False, drop_last=False) for tup in train_datasets]
    dev_dataset_loader = [torch.utils.data.DataLoader(
        tup, 64, shuffle=False, drop_last=False) for tup in dev_datasets]
    test_dataset_loader = [torch.utils.data.DataLoader(
        tup, 64, shuffle=False, drop_last=False) for tup in test_datasets]

    # 重写金标文件
    REWRITE = args.rewrite
    print('Writing gold file...')
    train_gold = write_gold_file(train_dataset_loader, args.train_file, idx2label, idx2feature, 'train_gold', REWRITE)
    dev_gold = write_gold_file(dev_dataset_loader, args.train_file, idx2label, idx2feature, 'dev_gold', REWRITE)
    test_gold = write_gold_file(test_dataset_loader, args.train_file, idx2label, idx2feature, 'test_gold', REWRITE)

    # 加载模型
    print('Loading model...')
    model = BiLSTM_CRF_L(len(feature2idx), label2idx, args.embedding_dim, args.hidden_size, args.layers, args.drop_out)
    model.to(device)

    # 加载模型参数
    if checkpoint_file is not None:
        print('Load state dict...')
        model.load_state_dict(checkpoint_file['state_dict'])
    else:
        print('Rand initing...')
        model.init_uniform()

    # 加载优化器以及参数
    if args.update == 'sgd':
        print('Loading Sgd optimizer...')
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        print('Loading Adam optimizer...')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if checkpoint_file is not None and args.load_opt:
        args.load_opt = args.load_opt.strip()
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    tot_iters = sum(map(lambda t: len(t), train_dataset_loader))
    try:
        # 部分早期模型没有这两个属性
        best_F1 = checkpoint_file['best_F1']
        best_acc = checkpoint_file['best_acc']
        print('Best F1: %f best_acc: %f' % (best_F1, best_acc))
    except:
        best_F1 = float('-inf')
        best_acc = float('-inf')
    start_time = time.time()
    end_epoch = args.epoch
    epoch_list = range(args.start_epoch + 1, end_epoch)
    patience_count = 0

    # 在dev上的最原始效果
    print('Origin performance')
    tmp_file = predict_write(model, dev_dataset_loader, idx2feature, idx2label, device)
    dev_R, dev_P, dev_F1 = evaluate_with_perl(dev_gold, tmp_file, args.log_path, 0)

    for epoch_idx, args.start_epoch in enumerate(epoch_list):
        epoch_start_time = time.time()
        epoch_loss = 0
        loss = 0
        model.train()

        for i, batch in enumerate(
                itertools.chain.from_iterable(train_dataset_loader)):
            model.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            features, labels, masks = batch
            batch_size = features.shape[0]
            features_v, labels_v, masks_v = features.transpose(0, 1), labels.transpose(0, 1), masks.transpose(0, 1)
            loss = model.neg_log_likelihood(features_v, labels_v, masks_v)

            print('loss: {:.4f}'.format(loss.item()), ' {:} / {:} iteration '.format(i + 1, tot_iters),
                  ' {:} / {:} epoch'.format(args.start_epoch, end_epoch))

            loss.backward()
            # 不梯度剪裁很容易出现梯度爆炸梯度消失！
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            epoch_loss += loss.item()

        # 调整学习率
        adjust_learning_rate(
            optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        epoch_loss /= tot_iters
        print('Epoch average loss %f, takes %f s' % (epoch_loss, time.time() - epoch_start_time))

        print('Dev result')
        # 这个效率太低了...
        # dev_P, dev_R, dev_F1, dev_ER = evaluate(model, dev_dataset_loader, idx2feature, idx2label, device,
        #                                         args.log_path)
        tmp_file = predict_write(model, dev_dataset_loader, idx2feature, idx2label, device)
        dev_R, dev_P, dev_F1 = evaluate_with_perl(dev_gold, tmp_file, args.log_path, args.start_epoch, loss=epoch_loss, dev=True)

        # tmp_file = predict_write(model, train_dataset_loader, idx2feature, idx2label, device)
        # train_R, train_P, train_F1 = evaluate_with_perl(train_gold, tmp_file, args.log_path, args.start_epoch, dev=False)

        states = {
            'epoch': args.start_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'feature2idx': feature2idx,
            'label2idx': label2idx,
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'layers': args.layers,
            'drop_out': args.drop_out,
            'best_F1': best_F1,
            'best_acc': best_acc
        }

        save_file_path = os.path.join(args.checkpoint_dir, 'current.pth')
        print('Saving model to ', save_file_path)
        torch.save(states, save_file_path)

        if dev_F1 > best_F1:
            # 保存模型
            patience_count = 0

            best_F1 = dev_F1
            best_acc = best_F1

            save_file_path = os.path.join(args.checkpoint_dir, 'best.pth')
            print('Saving model to ', save_file_path)

            torch.save(states, save_file_path)
        else:
            patience_count += 1
            print(patience_count)
            if patience_count == args.patience:
                print('Stop training early')
                break

    print('Final result : Acc : %{:.4f} , F1 : %{:.4f}'.format(best_acc, best_F1))
