import matplotlib.pyplot as plt
import re

log_paths = ['./checkpoint/128_128_128_0.30_1_adam_0.010/log',
             './checkpoint/128_128_128_0.30_1_sgd_0.200/log',
             './checkpoint/128_128_128_0.30_1_adam_0.050/log',
             './checkpoint/128_128_128_0.30_1_adam_0.100/log',
             ]
labels = [
    'adam lr = 0.01',
    'sgd lr = 0.2',
    'adam lr = 0.05',
    'adam lr =  0.1',
]


def plot(prefix='Epoch loss', length=-1, title='Batch - Loss'):

    for log_path, label in zip(log_paths, labels):
        scores = []
        with open(log_path, encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(prefix):
                    scores.append(float(re.split(r'[; \s\t]\s*', line.strip())[-1]))
        if log_path == './checkpoint/32_128_300_0.55_1/log':
            scores = [scores[0]] + scores[1::2]
        if log_path == './checkpoint/128_128_128_0.55_1/log':
            scores = scores[1:]
        if length > 0:
            cur_len = min(len(scores), length)
        else:
            cur_len = len(scores)
        plt.plot(range(cur_len), scores[:cur_len], label=label)

    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plot(prefix='=== F MEASURE', title='Optimizer method - F1', length=40)
    plot(prefix='Epoch loss', title='Optimizer method - Loss', length=40)