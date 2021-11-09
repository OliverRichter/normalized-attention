import numpy as np
import tensorflow as tf
import functools


def generator(batch_creator):
    """Simple wrapper to turn a batch creation function into an infinite generator"""
    @functools.wraps(batch_creator)
    def generator_wrapper(*args, **kwargs):
        while True:
            yield batch_creator(*args, **kwargs)
    return generator_wrapper


@generator
def create_argmin_first_argmax_batch(vocabulary_size=100, batch_size=32, sequence_length=128, case_bias=0):
    x = np.random.randint(0, vocabulary_size, (batch_size, sequence_length))
    for _ in range(-case_bias):
        for idx, data_point in enumerate(x):
            if 64 in data_point and np.random.rand() > 0.5:
                while 64 in data_point:
                    data_point[np.where(data_point == 64)[0][0]] = np.random.randint(0, vocabulary_size)
                x[idx] = data_point
    for _ in range(case_bias):
        for idx, data_point in enumerate(x):
            if 64 not in data_point and np.random.rand() > 0.5:
                x[idx, np.random.randint(sequence_length)] = 64
    y = np.zeros_like(x)
    for idx, data_point in enumerate(x):
        if 64 in data_point:
            y[idx, np.argmin(data_point)] = 1.0
        elif 50 in data_point:
            y[idx, 0] = 1.0
        else:
            y[idx, np.argmax(data_point)] = 1.0
    return x, y


def create_argmin_first_argmax_equalized_dataset(vocabulary_size=100, dataset_size=3000, sequence_length=128):
    """
    Creates a dataset for the argmin-first-argmax task that has 1/3 of the datapoints in the 'argmin' case,
    1/3 in the 'first' case and 1/3 in the 'argmax' case.
    :param vocabulary_size: The size of the vocabulary. Should be bigger than 64, as the 'argmin' case requires
    token 64 to be present.
    :param dataset_size: The overall size of the dataset created, i.e., 3 times the number of datapoints per major case
    :param sequence_length: The number of tokens from the vocabulary in each datapoint.
    :return: data points, labels, a multi hot encoding of which cases the data points fall into and a list of short
    names describing the 3 major (argmin, first, argmax) cases and 4 minor cases (separate cases are added to track when
    major cases collide, e.g., the first token happens to be the argmin).
    """
    assert vocabulary_size > 64
    x = np.random.randint(0, vocabulary_size, (dataset_size, sequence_length))
    y = np.zeros_like(x)
    case = np.zeros((dataset_size, 7))
    case_strings = ['64', '64 & argmin=first', '50', '50 & argmin=first',
                    '50 & argmax=first', 'else', 'else & argmax=first']
    for idx, data_point in enumerate(x):
        if 64 in data_point:
            if np.sum(case[:, 0]) >= dataset_size / 3:
                while 64 in data_point:
                    data_point[np.where(data_point == 64)[0][0]] = np.random.randint(0, vocabulary_size)
                x[idx] = data_point
            else:
                y[idx, np.argmin(data_point)] = 1.0
                case[idx, 0] = 1.0
                if np.argmin(data_point) == 0:
                    case[idx, 1] = 1.0
        if 64 not in data_point and 50 in data_point:
            if np.sum(case[:, 2]) >= dataset_size / 3:
                while 64 in data_point or 50 in data_point:
                    if 50 in data_point:
                        data_point[np.where(data_point == 50)[0][0]] = np.random.randint(0, vocabulary_size)
                    else:
                        data_point[np.where(data_point == 64)[0][0]] = np.random.randint(0, vocabulary_size)
                x[idx] = data_point
            else:
                y[idx, 0] = 1.0
                case[idx, 2] = 1.0
                if np.argmin(data_point) == 0:
                    case[idx, 3] = 1.0
                if np.argmax(data_point) == 0:
                    case[idx, 4] = 1.0
        if 64 not in data_point and 50 not in data_point:
            if np.sum(case[:, 5]) >= dataset_size / 3:
                if np.sum(case[:, 2]) < dataset_size / 3:
                    data_point[np.random.randint(0, sequence_length)] = 50
                    y[idx, 0] = 1.0
                    case[idx, 2] = 1.0
                    if np.argmin(data_point) == 0:
                        case[idx, 3] = 1.0
                    if np.argmax(data_point) == 0:
                        case[idx, 4] = 1.0
                else:
                    data_point[np.random.randint(0, sequence_length)] = 64
                    y[idx, np.argmin(data_point)] = 1.0
                    case[idx, 0] = 1.0
                    if np.argmin(data_point) == 0:
                        case[idx, 1] = 1.0
                x[idx] = data_point
            else:
                y[idx, np.argmax(data_point)] = 1.0
                case[idx, 5] = 1.0
                if np.argmax(data_point) == 0:
                    case[idx, 6] = 1.0
    return x, y, case, case_strings


def eval_cases(model, inputs, labels, cases, case_strings):
    predictions = model.predict(inputs)
    correct_predictions = np.equal(np.argmax(predictions, -1), np.argmax(labels, -1))
    correct_per_case = np.sum(correct_predictions[:, np.newaxis] * cases, axis=0, dtype=np.int)
    case_occurances = np.sum(cases, axis=0, dtype=np.int)
    if len(case_occurances.shape) > 1:
        correct_per_case = np.mean(correct_per_case, axis=-1)
        case_occurances = np.sum(case_occurances, axis=-1)
    for case in range(len(case_occurances)):
        print('Case ' + case_strings[case] + ':')
        print(str(correct_per_case[case]) + ' / ' + str(case_occurances[case]) + ' = '
              + str(float(correct_per_case[case]) / float(case_occurances[case] + 1e-20)))
    return correct_per_case, case_occurances


class CaseEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, results_buffer, indices, vocab_size, sequence_length, task, validate, **kwargs):
        super(CaseEvaluationCallback, self).__init__(**kwargs)
        self.results = results_buffer
        self.indices = indices
        self.vocab_size = vocab_size
        self.seq_len = sequence_length
        self.validate = validate
        if task == 'cases':
            self.equalized_dataset = create_argmin_first_argmax_equalized_dataset
            self.num_cases = 7
        else:
            self.equalized_dataset = create_local_global_equalized_dataset
            self.num_cases = 2

    def on_epoch_end(self, epoch, logs=None):
        x, y, cases, case_str = self.equalized_dataset(self.vocab_size, 3000, self.seq_len)
        print('\nCase accurracy: ')
        case_results = eval_cases(self.model, x, y, cases, case_str)
        offset = 4
        for idx in range(self.num_cases):
            self.results[epoch][self.indices[1]][self.indices[0]][idx + offset].append(case_results[0][idx])
        offset += self.num_cases
        for idx in range(self.num_cases):
            self.results[epoch][self.indices[1]][self.indices[0]][idx + offset].append(case_results[1][idx])
        if self.validate:
            x, y, cases, case_str = self.equalized_dataset(self.vocab_size, 3000, self.seq_len // 2)
            print('Validation set case accurracy: ')
            validation_case_results = eval_cases(self.model, x, y, cases, case_str)
            offset += self.num_cases
            for idx in range(self.num_cases):
                self.results[epoch][self.indices[1]][self.indices[0]][idx + offset].append(
                    validation_case_results[0][idx])
            offset += self.num_cases
            for idx in range(self.num_cases):
                self.results[epoch][self.indices[1]][self.indices[0]][idx + offset].append(
                    validation_case_results[1][idx])


@generator
def create_mode_batch(vocabulary_size=10, batch_size=32, sequence_length=128):
    x = np.random.randint(0, vocabulary_size, (batch_size, sequence_length))
    from scipy.stats import mode
    y = np.reshape(mode(x, axis=1)[0], (-1,))
    return x, y


@generator
def create_local_global_batch(vocabulary_size=100, batch_size=32, sequence_length=128, case_bias=0):
    x = np.random.randint(0, vocabulary_size, (batch_size, sequence_length))
    for _ in range(-case_bias):
        for idx, data_point in enumerate(x):
            if 64 in data_point and np.random.rand() > 0.5:
                while 64 in data_point:
                    data_point[np.where(data_point == 64)[0][0]] = np.random.randint(0, vocabulary_size)
                x[idx] = data_point
    for _ in range(case_bias):
        for idx, data_point in enumerate(x):
            if 64 not in data_point and np.random.rand() > 0.5:
                x[idx, np.random.randint(sequence_length)] = 64
    y = np.zeros((batch_size, sequence_length, sequence_length))
    for idx, data_point in enumerate(x):
        if 64 in data_point:
            y[idx, np.arange(sequence_length), np.arange(sequence_length)] = 1.0
        else:
            y[idx, :, np.argmin(data_point)] = 1.0
    return x, y


def create_local_global_equalized_dataset(vocabulary_size=100, dataset_size=3000, sequence_length=128):
    assert vocabulary_size > 64
    x = np.random.randint(0, vocabulary_size, (dataset_size, sequence_length))
    y = np.zeros((dataset_size, sequence_length, sequence_length))
    case = np.zeros((dataset_size, 2, 1))
    case_strings = ['64', 'else']
    for idx, data_point in enumerate(x):
        if 64 in data_point:
            if np.sum(case[:, 0]) >= dataset_size / 2:
                while 64 in data_point:
                    data_point[np.where(data_point == 64)[0][0]] = np.random.randint(0, vocabulary_size)
                x[idx] = data_point
            else:
                y[idx, np.arange(sequence_length), np.arange(sequence_length)] = 1.0
                case[idx, 0] = 1.0
        if 64 not in data_point:
            if np.sum(case[:, 1]) >= dataset_size / 2:
                data_point[np.random.randint(0, sequence_length)] = 64
                y[idx, np.arange(sequence_length), np.arange(sequence_length)] = 1.0
                case[idx, 0] = 1.0
                x[idx] = data_point
            else:
                y[idx, :, np.argmin(data_point)] = 1.0
                case[idx, 1] = 1.0
    return x, y, case, case_strings




