import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def mean_norm(sequence):
    return np.mean(np.linalg.norm(sequence, axis=-1))


def softmax(logs):
    return np.exp(logs) / np.sum(np.exp(logs), axis=-1, keepdims=True)


def normalize(logs, axis=-1, epsilon=0.001):
    if logs.shape[-1] == 1:
        return np.ones_like(logs)
    return (logs - np.mean(logs, axis=axis, keepdims=True)) / (np.std(logs, axis=axis, keepdims=True) + epsilon)


def plot(init_samples=1024*16, max_sequence_length_exp=12, model_dim=128):
    std_results = {'softmax attention': [],
                   'mean pooling': [],
                   'sum pooling': [],
                   'max pooling': [],
                   'normalized': []}
    norm_results = {'softmax attention': [],
                    'mean pooling': [],
                    'sum pooling': [],
                    'max pooling': [],
                    'normalized': []}
    sequence_lengths = [2 ** sequence_length_exp for sequence_length_exp in range(max_sequence_length_exp)]
    for sequence_length in sequence_lengths:
        samples = init_samples // sequence_length
        values = np.random.normal(size=[samples, sequence_length, model_dim])
        keys = np.random.normal(size=[samples, sequence_length, model_dim])
        querries = np.random.normal(size=[samples, model_dim, sequence_length])
        logits = np.matmul(keys, querries) / np.sqrt(model_dim)

        attention = softmax(logits)
        out = {'softmax attention': np.matmul(attention, values),
               'mean pooling': np.mean(values, axis=1, keepdims=True),
               'sum pooling': np.sum(values, axis=1, keepdims=True),
               'max pooling': np.max(values, axis=1, keepdims=True),
               'normalized': normalize(np.sum(values, axis=1, keepdims=True))}

        for key in std_results.keys():
            std_results[key].append(np.std(out[key]))
            norm_results[key].append(mean_norm(out[key]))

    plt.figure('Standard Deviation')
    for result in std_results.values():
        plt.loglog(sequence_lengths, result)
    plt.ylabel('Standard Deviation')
    plt.xlabel('Sequence Length')

    tikzplotlib.save('std_scaling.tex')

    plt.figure('Norm')
    for result in norm_results.values():
        plt.loglog(sequence_lengths, result)
    plt.ylabel('Norm')
    plt.xlabel('Sequence Length')
    plt.legend(norm_results.keys(), loc='lower left')

    tikzplotlib.save('norm_scaling.tex')


plot()
plt.show()
