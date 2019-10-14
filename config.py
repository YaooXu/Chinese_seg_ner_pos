import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Learning with BLSTM-CRF')

    parser.add_argument('--rand_embedding', action='store_true',
                        help='random initialize word embedding')
    parser.add_argument('--embedding_file', default='',
                        help='path to pre-trained embedding')

    parser.add_argument(
        '--train_file', default='./data/renmin98/little_train', help='path to training file')

    parser.add_argument(
        '--dev_ratio', type=float, default='0.2', help='the ratio of dev data in all data')

    parser.add_argument(
        '--test_file', default='./data/renmin98/test', help='path to test file')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size (10)')
    parser.add_argument('--unk', default='unk',
                        help='unknow-token in pre-trained embedding')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/',
                        help='path to checkpoint prefix')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='hidden dimension')
    parser.add_argument('--drop_out', type=float,
                        default=0.55, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200,
                        help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int,
                        default=0, help='start epoch idx')
    parser.add_argument('--embedding_dim', type=int,
                        default=100, help='dimension for word embedding')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.015,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05,
                        help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false',
                        help='fine tune pre-trained embedding dictionary')
    parser.add_argument('--load_check_point', default='',
                        help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true',
                        help='load optimizer from ')
    parser.add_argument(
        '--update', choices=['sgd', 'adam'], default='sgd', help='optimizer method')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float,
                        default=5.0, help='grad clip at')
    parser.add_argument('--mini_count', type=float, default=5,
                        help='thresholds to replace rare words with <unk>')
    parser.add_argument('--patience', type=int, default=15,
                        help='patience for early stop')
    parser.add_argument('--shrink_embedding', action='store_true',
                        help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')

    args = parser.parse_args()
    return args