import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Learning with BLSTM-CRF')

    parser.add_argument('--rand_embedding', action='store_true',
                        help='random initialize word embedding')
    parser.add_argument('--embedding_file', default='',
                        help='path to pre-trained embedding')

    parser.add_argument(
        '--train_file', default='./data/little_renmin98/train', help='path to training file')
    parser.add_argument(
        '--dev_file', default='./data/little_renmin98/dev', help='path to training file')
    parser.add_argument(
        '--test_file', default='./data/little_renmin98/test', help='path to test file')
    parser.add_argument('--rewrite', type=int, default=0,
                        help='whether to rewrite gold file')

    parser.add_argument('--gpu', type=int, default=3,
                        help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size (10)')
    parser.add_argument('--unk', default='unk',
                        help='unknow-token in pre-trained embedding')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/',
                        help='path to checkpoint prefix')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='hidden dimension')
    parser.add_argument('--drop_out', type=float,
                        default=0.55, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200,
                        help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int,
                        default=0, help='start epoch idx')
    parser.add_argument('--embedding_dim', type=int,
                        default=128, help='dimension for word embedding')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of LSTM layers')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05,
                        help='decay ratio of learning rate')
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
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stop')
    parser.add_argument('--large', type=int, default=1,
                        help='whether to use large model')
    args = parser.parse_args()
    return args