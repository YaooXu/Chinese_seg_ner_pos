from utils import read_corpus


def write():
    tgt_file = './dev_order_gold'
    dev_file = r'/home/yxu/Seg_ner_pos/data/little_renmin/dev'

    with open(dev_file, encoding='utf8') as f:
        lines = f.readlines()

    features, labels = read_corpus(lines)

    with open(tgt_file, 'w', encoding='utf8') as f:
        for feature, label in zip(features, labels):
            line = ''
            for i in range(len(label)):
                if label[i][0] in ('B', 'S') and i != 0:
                    line += ' ' + feature[i]
                else:
                    line += feature[i]
            f.write(line + '\n')
    return tgt_file


if __name__ == '__main__':
    write()
