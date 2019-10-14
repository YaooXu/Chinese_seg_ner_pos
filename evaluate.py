import itertools
import tqdm

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


def evaluate(model, dataset_loader, idx2feature, idx2label, device):
    r"""
    计算数据集上的F1, P, R值

    :return: F1, accuracy
    """
    num_sentence = 0
    num_right_sentence = 0

    num_all_word = 0
    num_right_word = 0
    num_error_word = 0

    model.eval()
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
                if j + 1 == length or masks[i][j + 1] == 0:
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

    return P, R, F1, ER