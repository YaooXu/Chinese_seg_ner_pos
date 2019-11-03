import itertools
import tqdm
import subprocess


def add_to_seq(seq, feature, label):
    if label[0] == 'B':
        seq.append(feature)
    elif label[0] == 'I':
        if len(seq) > 0:
            seq[-1] += feature
        else:
            seq.append(feature)
    elif label[0] == 'S':
        seq.append(feature)


def cal(predict_seq, right_seq):
    r"""
    :return: 分词正确的个数，错误个数，总单词个数，该句子是否分词正确
    """
    # flag 表示全部正确
    flag = 1
    num_right_word = 0

    if len(predict_seq) != len(right_seq):
        flag = 0

    for feat in predict_seq:
        if feat in right_seq:
            num_right_word += 1
        else:
            flag = 0

    return num_right_word, len(predict_seq) - num_right_word, len(right_seq), flag


def evaluate(model, dataset_loader, idx2feature, idx2label, device, log_file):
    r"""
    计算数据集上的F1, P, R值

    :return: F1, accuracy
    """
    model.eval()

    num_sentence = 0
    num_right_sentence = 0

    num_all_word = 0
    num_right_word = 0
    num_error_word = 0

    for _, batch in tqdm.tqdm(enumerate(itertools.chain.from_iterable(dataset_loader))):
        batch = tuple(t.to(device) for t in batch)
        features, labels, masks = batch

        scores, paths = model(features, masks)

        num_sentence += features.shape[0]
        length = features.shape[1]
        for i, (sentence, label) in enumerate(zip(features, labels)):
            predict_seq = []
            right_seq = []
            for j, tensor_feat in enumerate(sentence):
                if j == length or masks[i][j] == 0:
                    # 会有一个<eof>标志
                    nums = cal(predict_seq, right_seq)

                    num_right_word += nums[0]
                    num_error_word += nums[1]
                    num_all_word += nums[2]
                    num_right_sentence += nums[3]

                    break
                else:
                    feature = idx2feature[tensor_feat.item()]
                    predict_label = idx2label[paths[i][j].item()]
                    right_label = idx2label[label[j].item()]

                    add_to_seq(predict_seq, feature, predict_label)
                    add_to_seq(right_seq, feature, right_label)

    P = num_right_word / (num_error_word + num_right_word)
    R = num_right_word / (num_all_word)
    F1 = (2 * P * R) / (P + R)
    ER = num_error_word / num_all_word

    print(
        '标准词数：%d个，词数正确率：%f个，词数错误率：%f' % (num_all_word, num_right_word / num_all_word, num_error_word / num_all_word))
    print('标准行数：%d，行数正确率：%f' % (num_sentence, num_right_sentence / num_sentence))
    print('Recall: %f' % (R))
    print('Precision: %f' % (P))
    print('F1 MEASURE: %f' % (F1))
    print('ERR RATE: %f' % (ER))

    with open(log_file, 'a') as f:
        f.write(
            '标准词数：%d个，词数正确率：%f个，词数错误率：%f\n' % (
                num_all_word, num_right_word / num_all_word, num_error_word / num_all_word))
        f.write('标准行数：%d，行数正确率：%f\n' % (num_sentence, num_right_sentence / num_sentence))
        f.write('Recall: %f\n' % (R))
        f.write('Precision: %f\n' % (P))
        f.write('F1 MEASURE: %f\n' % (F1))
        f.write('ERR RATE: %f\n\n\n' % (ER))

    return P, R, F1, ER


def evaluate_with_perl(gold_file, predict_file, log, epoch, loss=None, dev=True):
    r"""
    这个效率高

    :param gold_file:
    :param predict_file:
    :return:
    """
    perl_path = './icwb2-data/scripts/score'
    word_list = './icwb2-data/gold/pku_training_words.utf8'
    p = subprocess.Popen(['perl', perl_path, word_list, gold_file, predict_file], stdout=subprocess.PIPE)
    output = p.stdout.read()
    output = output.decode(encoding='utf8')
    outputs = output.split('\n')
    p.kill()
    res = outputs[-15:]
    dev_R, dev_P, dev_F1 = float(res[-8].split('\t')[-1]), float(res[-7].split('\t')[-1]), float(
        res[-6].split('\t')[-1])

    with open(log, 'a') as f:
        f.write('EPOCH : %d\n' % epoch)

        if dev:
            f.write('Dev\n')
        else:
            f.write('Train\n')

        if loss is not None:
            f.write('Epoch loss : %f\n' % loss)

        for j in res:
            print(j)
            f.write(j + '\n')

    return dev_R, dev_P, dev_F1


def predict_write(model, dataset_loader, idx2feature, idx2label, device, tmp_file='./tmp'):
    r"""
    返回一个临时的预测文件

    :param model:
    :param dataset_loader:
    :param idx2feature:
    :param idx2label:
    :param device:
    :param tmp_file:
    :return:
    """
    # !!
    model.eval()

    with open(tmp_file, 'w') as f:
        for _, batch in enumerate(itertools.chain.from_iterable(dataset_loader)):
            batch = tuple(t.to(device) for t in batch)
            features, labels, masks = batch
            features_v, labels_v, masks_v = features.transpose(0, 1), labels.transpose(0, 1), masks.transpose(0, 1)
            scores, predict_labels = model.predict(features_v, masks_v)
            for j in range(labels.shape[0]):
                feature, predict_label, mask = features[j], predict_labels[j], masks[j]
                line = ''
                length = predict_label.shape[0]
                for k in range(length):
                    # # TODO:DEBUG
                    # if k == 0:
                    #     continue
                    if k + 1 == length or mask[k + 1].item() == 0:
                        f.write(line + '\n')
                        break
                    else:
                        if idx2label[predict_label[k].item()][0] in ('B', 'S') and k != 0:
                            line += ' ' + idx2feature[feature[k].item()]
                        else:
                            line += idx2feature[feature[k].item()]
    return tmp_file


def read_line(f):
    '''
        读取一行，并清洗空格和换行
    '''
    line = f.readline()
    return line.strip()


def evaluate_by_file(real_text_file, pred_text_file, prf_file, epoch):
    file_gold = open(real_text_file, 'r', encoding='utf8')
    file_tag = open(pred_text_file, 'r', encoding='utf8')

    line1 = read_line(file_gold)
    N_count = 0  # 将正类分为正或者将正类分为负
    e_count = 0  # 将负类分为正
    c_count = 0  # 正类分为正
    e_line_count = 0
    c_line_count = 0

    while line1:
        line2 = read_line(file_tag)

        list1 = line1.split(' ')
        list2 = line2.split(' ')

        count1 = len(list1)  # 标准分词数
        N_count += count1
        if line1 == line2:
            c_line_count += 1  # 分对的行数
            c_count += count1  # 分对的词数
        else:
            e_line_count += 1
            count2 = len(list2)

            arr1 = []
            arr2 = []

            pos = 0
            for w in list1:
                arr1.append(tuple([pos, pos + len(w)]))  # list1中各个单词的起始位置
                pos += len(w)

            pos = 0
            for w in list2:
                arr2.append(tuple([pos, pos + len(w)]))  # list2中各个单词的起始位置
                pos += len(w)

            for tp in arr2:
                if tp in arr1:
                    c_count += 1
                else:
                    e_count += 1

        line1 = read_line(file_gold)

    R = float(c_count) / N_count
    P = float(c_count) / (c_count + e_count)
    F = 2. * P * R / (P + R)
    ER = 1. * e_count / N_count

    print("result:")
    print('标准词数：%d个，词数正确率：%f个，词数错误率：%f' % (N_count, c_count / N_count, e_count / N_count))
    print('标准行数：%d，行数正确率：%f，行数错误率：%f' % (c_line_count + e_line_count, c_line_count / (c_line_count + e_line_count),
                                         e_line_count / (c_line_count + e_line_count)))
    print('Recall: %f' % (R))
    print('Precision: %f' % (P))
    print('F MEASURE: %f' % (F))
    print('ERR RATE: %f' % (ER))

    # print P,R,F

    f = open(prf_file, 'a', encoding='utf-8')
    f.write('result-(epoch:%s):\n' % epoch)
    f.write('标准词数：%d，词数正确率：%f，词数错误率：%f \n' % (N_count, c_count / N_count, e_count / N_count))
    f.write('标准行数：%d，行数正确率：%f，行数错误率：%f \n' % (c_line_count + e_line_count, c_line_count / (c_line_count + e_line_count),
                                              e_line_count / (c_line_count + e_line_count)))
    f.write('Recall: %f\n' % (R))
    f.write('Precision: %f\n' % (P))
    f.write('F MEASURE: %f\n' % (F))
    f.write('ERR RATE: %f\n' % (ER))
    f.write('====================================\n')

    return F
