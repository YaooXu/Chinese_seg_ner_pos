import os
import re
import argparse
from tqdm import tqdm

# 把98年人民日报的原始数据集转化为CoNLL 2003的数据格式
# 句子之间以空行分割开来
# 文件之间以如下字符串分隔
# -DOCSTART- -X- -X- -X- O

"""
-DOCSTART- -X- -X- -X- O

Pierre NNP
Vinken NNP
, ,
61 CD
years NNS
old JJ
, ,
will MD
join VB
the DT
board NN
as IN
a DT
nonexecutive JJ
director NN
Nov. NNP
29 CD
"""


def splite_in_middle(line, splite_symbol='。'):
    """
    用中间的句号把一个长句子分割开来

    :param line: string,需要分割的长句子
    :param splite_symbol: 分隔符,默认为'。'
    :return: list, 包含切割后的两个字句
    """
    print(line)
    idxes = [i for i in range(len(line)) if line[i] == splite_symbol]
    if (len(idxes) == 0):
        # 一个没有句号的超长句子，尝试用逗号分割
        return splite_in_middle(line, '，')
    else:
        idx = idxes[len(idxes) // 2]
        return [line[:idx], line[idx + 1:]]


def try_delete(file):
    try:
        os.remove(file)
        print('delete old %s file' % file)
    except:
        print('build new %s' % file)


def convert_to_bis(source_dir, target_path, test_ratio=-1, only_seg=False, max_len=-1):
    r"""

    :param source_dir:
    :param target_path:
    :param test_ratio: -1则表示不切分测试集
    :param only_seg:
    :param max_len: 如果为-1则不进行句子截断
    :return:
    """
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    print("Converting...")
    train_file = os.path.join(target_path, 'train')
    test_file = os.path.join(target_path, 'test')
    dev_file = os.path.join(target_path, 'dev')

    try_delete(train_file)
    try_delete(test_file)
    try_delete(dev_file)

    for root, dirs, files in os.walk(source_dir):
        tgt_dir = target_path + root[len(source_dir):]
        print(tgt_dir)

        for index, file_name in tqdm(enumerate(files)):
            if file_name[:4] == 'hand':
                continue

            file = os.path.join(root, file_name)
            # 先预处理一遍
            handled_file = originHandle(file)

            with open(handled_file, 'r', encoding='utf-8') as f:
                BISes = process_lines(f.readlines(), only_seg, max_len)

            if test_ratio > 0:
                len_BISes = len(BISes)
                num_train = int(len_BISes * (1.0 - test_ratio * 2))
                num_test = int(len_BISes * test_ratio)
                _write_BISes(BISes[:num_train], train_file, write_mode='a')
                _write_BISes(BISes[num_train:num_train + num_test], test_file, write_mode='a')
                _write_BISes(BISes[-num_test:], dev_file, write_mode='a')
            else:
                _write_BISes(BISes, train_file, write_mode='a')
    print('Finish')


def _write_BISes(BISes, path, write_mode):
    r"""
    把传递过来的list以CoNLL 2003数据集的格式写到文件

    :param BISes: 处理过后的文本
    :param path: 输出文件路径
    :param write_mode: 写文件格式
    :return:
    """
    DOC_SPLITE_LINE = "-DOCSTART- -X - -X - -X - O\n\n"

    with open(path, mode=write_mode, encoding='UTF-8') as f:
        f.write(DOC_SPLITE_LINE)

        for line in BISes:
            for char, tag in line:
                f.write(char + " " + tag + '\n')
            f.write('\n')


def process_lines(lines: list, only_seg=False, max_len=-1):
    """
    :param lines:
    :param only_seg:
    :return: BISes: list
        每个元素均为一个list，代表原始数据集的一行
    """

    BISes = []
    for line in lines:
        line, _ = re.subn('\n', '', line)
        if line == '' or line == '\n':
            continue

        # \s+, 匹配任意长度空格
        words = re.split(r'\s+', line)
        BISes.append(_tag(words, only_seg, max_len))

    return BISes


def _tag(words, only_seg=False, max_len=-1):
    """
    给指定的一行文本打上BIS标签，同时限定最大长度，预处理之后其实就不再需要处理[]了

    :param words: 把原行以空格分隔产生的列表
    :return: BIS : List
        其中每个元素都是一个tuple (word, tag)
    """
    BIS = []
    tot_len = 0

    for word in words:
        # 最终要以[]的整体label作为每个词语的label
        if max_len > 0 and tot_len >= max_len:
            break

        tokens = word.split('/')
        if len(tokens) == 2:
            word, pos = tokens
        elif len(tokens) == 3:
            # ex: 基金 / n] / nz ...
            # 腾空而起 l %
            word, middle, pos = tokens
        else:
            # TODO: 是否有特殊情况
            continue

        word = list(word)
        if pos == '%':
            pos = 'l'
        pos = pos.upper()

        if len(word) == 0:
            continue

        if not only_seg:
            if len(word) == 1:
                BIS.append((word[0], 'S-' + pos))
                tot_len += 1
                if tot_len >= max_len:
                    break
            else:
                for i, char in enumerate(word):
                    if i == 0:
                        BIS.append((char, 'B-' + pos))
                    else:
                        BIS.append((char, 'I-' + pos))

                    tot_len += 1
                    if tot_len >= max_len:
                        break
        else:
            # 只保留BIS
            if len(word) == 1:
                BIS.append((word[0], 'S'))
                tot_len += 1
                if tot_len >= max_len:
                    break
            else:
                for i, char in enumerate(word):
                    if i == 0:
                        BIS.append((char, 'B'))
                    else:
                        BIS.append((char, 'I'))

                    tot_len += 1
                    if tot_len >= max_len:
                        break

    return BIS


def originHandle(file):
    r"""
    对原始文本第一次处理，包括去掉开头编号，人名合并，[]删除（使用总体label代替[]中的单独label)

    :param file: 原始数据文件
    :return: handled_file 处理完之后的文件名
    """
    root, file_name = os.path.split(file)
    handled_file = os.path.join(root, "handled_" + file_name)

    if not os.path.exists(handled_file):
        with open(file, 'r', encoding='UTF-8') as inp, open(handled_file, 'w', encoding='UTF-8') as outp:
            for line in inp.readlines():
                line = line.split('  ')
                i = 1
                while i < len(line) - 1:
                    if line[i][0] == '[':
                        outp.write(line[i].split('/')[0][1:])
                        i += 1
                        while i < len(line) - 1 and line[i].find(']') == -1:
                            if line[i] != '':
                                outp.write(line[i].split('/')[0])
                            i += 1
                        # 存在]l...
                        idx = line[i].find(']')
                        outp.write(line[i].split('/')[0].strip() +
                                   '/' + line[i][idx + 1:] + ' ')

                    elif line[i].split('/')[1] == 'nr':
                        # 对姓名进行合并处理
                        word = line[i].split('/')[0]
                        i += 1
                        if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                            outp.write(word + line[i].split('/')[0] + '/nr ')
                        else:
                            outp.write(word + '/nr ')
                            continue
                    else:
                        outp.write(line[i] + ' ')
                    i += 1
                outp.write('\n')
    return handled_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将使用词性标注的文件转换为用BIS分块标记的文件。")
    parser.add_argument("--corups_dir", type=str,
                        help="指定存放语料库的文件夹，程序将会递归查找目录下的文件。")
    parser.add_argument("--output_path", type=str,
                        default='.', help="指定标记好的文件的输出路径。")
    parser.add_argument("--test_ratio", type=float,
                        default='-1', help="从整体数据中分割出来的测试集比例")
    parser.add_argument("--max_len", type=int,
                        default=-1, help="数据集最长长度")
    parser.add_argument("--only_seg", action='store_true',
                        help="是否只保留BIS标签")
    args = parser.parse_args()

    convert_to_bis(args.corups_dir, args.output_path, args.test_ratio, args.only_seg, args.max_len)
