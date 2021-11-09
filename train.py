from models import create_model
from optimization import WarmUpThenLinDecaySchedule, ClipAdam
from task_generators import create_argmin_first_argmax_batch, create_mode_batch, create_local_global_batch,\
    CaseEvaluationCallback
import numpy as np
import tensorflow as tf
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cases', choices=['cases', 'mode', 'lg', 'ppi'],
                    help="The task to train on, can be either 'cases' for the case distinction/pin-pointing task or "
                         "'mode' for the mode finding task. Defaults to 'cases'.")
parser.add_argument('-f', '--first_token_output', action='store_true',
                    help='Flag to switch from an output taken across all tokens to the output taken from the first '
                         'token in the case distinction task.')
parser.add_argument('--variation', type=str, default='model_dimension',
                    help="Option to specify the hyperparamerer over which to sweep over. Can be abbreviated, "
                         "e.g., 'b' for 'batch_size'. Defaults to 'model_dimension'.")
parser.add_argument('-sv', '--skip_validation', action='store_true',
                    help='Skip validation on sequences of other length')
parser.add_argument('-t', '--training_runs', type=int, default=5,
                    help='Number of random seeds to use per hyper parameter combination. Defaults to 5.')
parser.add_argument('--model', type=str, default='NAP',
                    help='The architecture model to train. Defaults to NAP.')
parser.add_argument('-w', '--warmup', action='store_true',
                    help='Flag to include learning rate warm up.')
parser.add_argument('-c', '--clip_grads', action='store_true',
                    help='Flag to include gradient norm clipping.')
parser.add_argument('-ln', '--BERT_layer_norm_placement', action='store_true',
                    help='Reverts the placement and number of layer normalizations to the version proposed in BERT, '
                         'i.e., layer normalization is applied around the sum of layer output and skip connection and '
                         'not on hidden embeddings within the blocks.')
parser.add_argument('-no_gelu', '--no_non_linearity_in_attention_block', action='store_true',
                    help='Removes the gelu non-linearity on the hidden embeddings in the attention blocks.')
parser.add_argument('-xy', '--train_only_xy', type=tuple, default=None,
                    help='Train only the indexed hyperparameter combination.')

# Run order
parser.add_argument('-rev', '--reverse_run_order',
                    help='Start with the biggest model and smallest learning rate instead of the smallest model '
                         'and biggest learning rate. Ignored when random_run_order is provided.',
                    action='store_true')
parser.add_argument('-rro', '--random_run_order',
                    help='Randomizes the run order in which hyper parameter combinations are tried. Good for a first '
                         'glance, however, number of training runs per combination is now only given in expectation.',
                    action='store_true')

# Custom variation arguments
parser.add_argument('--var_min', type=int)
parser.add_argument('--var_range', type=int)
parser.add_argument('--var_base', type=float)

# Result location
parser.add_argument('--prefix', type=str, default='',
                    help='Prefix to specify the location where the training results should be stored.'
                         'Defaults to the current folder.')

args = parser.parse_args()
xy = None
if args.train_only_xy:
    xy = list(map(int, args.train_only_xy))
    assert len(xy) == 2

# Default configuration
params = {'layers': 2,
          'heads': 4,
          'model_dimension': 128,
          'initial_std_factor': 1,
          'max_sequence_length': 128,
          'case_bias': 0,
          'vocabulary_size': 100,
          'batch_size': 32,
          'learning_rate': 0.3 ** 7,
          'epochs': 32,
          'steps_per_epoch': 100,
          'validation_steps': 32,
          'dropout': 0.0,
          'L2_regularization': 0.0,
          'with_attention_mask': False}

experiment_name = ''
if args.prefix:
    if args.prefix[-1] == '/':
        experiment_name = args.prefix
    else:
        experiment_name = args.prefix + '/'
experiment_name += args.task + '/'

if args.task == 'cases':
    NUM_CASES = 7
    positional_embeddings = True

    def last_layer(contextual_embeddings, initializer, regularizer):
        if args.first_token_output:
            # project from the first token of size 'Model Dimension' to 'Max Sequence Length' outputs
            pre_logits = tf.keras.layers.Dense(params['max_sequence_length'],
                                               kernel_initializer=initializer,
                                               kernel_regularizer=regularizer)(contextual_embeddings[:, 0])
            # slice result in case that the actual sequence length is shorter than 'Max Sequence Length'
            logits = pre_logits[:, :tf.shape(contextual_embeddings)[1]]
        else:
            # project from every token to a 1 dimensional output
            logits = tf.keras.layers.Dense(1,
                                           kernel_initializer=initializer,
                                           kernel_regularizer=regularizer)(contextual_embeddings)[:, :, 0]
        return logits

    generator = create_argmin_first_argmax_batch
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.categorical_accuracy
    metric_name = 'categorical_accuracy'

    if args.first_token_output:
        experiment_name += 'first_token_output/'

elif args.task == 'lg':  # local vs. global task
    NUM_CASES = 2
    positional_embeddings = True

    def last_layer(contextual_embeddings, initializer, regularizer):
        # project from the first token of size 'Model Dimension' to 'Max Sequence Length' outputs
        pre_logits = tf.keras.layers.Dense(params['max_sequence_length'],
                                           kernel_initializer=initializer,
                                           kernel_regularizer=regularizer)(contextual_embeddings)
        # slice result in case that the actual sequence length is shorter than 'Max Sequence Length'
        logits = pre_logits[:, :, :tf.shape(contextual_embeddings)[1]]
        return logits

    generator = create_local_global_batch
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.categorical_accuracy
    metric_name = 'categorical_accuracy'

elif args.task == 'mode':
    NUM_CASES = 0
    params['vocabulary_size'] = 10  # update default value
    positional_embeddings = False

    def last_layer(contextual_embeddings, initializer, regularizer):
        return tf.keras.layers.Dense(params['vocabulary_size'],
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer)(contextual_embeddings[:, 0])

    generator = create_mode_batch
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.sparse_categorical_accuracy
    metric_name = 'sparse_categorical_accuracy'

elif args.task == 'ppi':
    NUM_CASES = 0
    params['layers'] = 3
    from ppi import create_and_run_model
    metric_name = 'accuracy'

else:
    raise NotImplementedError('Task not recognized')


if args.variation in ['d', 'dim', 'model_dimension']:
    variations = [{'name': 'model_dimension', 'min': 3, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_dim/'

elif args.variation in ['i', 'I', 'init', 'initialization', 'initial_std']:
    variations = [{'name': 'initial_std_factor', 'min': -3, 'range': 8, 'base': 4},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_init/'

elif args.variation in ['bias', 'case_bias']:
    variations = [{'name': 'case_bias', 'min': -3, 'range': 8, 'base': False},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_bias/'

elif args.variation in ['l', 'L', 'layers']:
    variations = [{'name': 'layers', 'min': 0, 'range': 7, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_layers/'

elif args.variation in ['h', 'H', 'm', 'M', 'heads']:
    variations = [{'name': 'heads', 'min': 0, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_heads/'

elif args.variation in ['s', 'S', 'seq', 'len', 'max_sequence_length']:
    variations = [{'name': 'max_sequence_length', 'min': 2, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_seq/'

elif args.variation in ['b', 'bs', 'batch', 'batch_size']:
    variations = [{'name': 'batch_size', 'min': 0, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_batch/'

elif args.variation in ['v', 'vs', 'vocab', 'vocabulary_size']:
    variations = [{'name': 'vocabulary_size', 'min': 1, 'range': 8, 'base': 2},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_vocab/'

elif args.variation in params.keys():
    if args.variation not in ['learning_rate', 'dropout', 'L2_regularization']:
        args.var_base = int(args.var_base)  # other hyperparameters only accept integer values
    variations = [{'name': args.variation, 'min': args.var_min, 'range': args.var_range, 'base': args.var_base},
                  {'name': 'learning_rate', 'min': 1, 'range': 10, 'base': 0.3}]
    experiment_name += 'var_' + args.variation.lower().replace(" ", "_") \
                       + '_' + str(args.var_min) + '_' + str(args.var_range) + '_' + str(args.var_base) + '/'

else:
    raise NotImplementedError('Variation not recognised')


try:
    os.makedirs(experiment_name)
    print("Directory ", experiment_name,  " created ")
except FileExistsError:
    print("Directory ", experiment_name,  " already exists, reusing this directory.")

if args.model in ['b', 'B', 'bert', 'BERT']:
    params.update({'model': 'BERT'})
    experiment_name += 'BERT'
    print("Choosing model 'BERT' overwrites command line arguments regarding learning rate warmup, gradient clipping,"
          "layer norm placement and non linearity in the attention block to the original BERT configuration. If you "
          "want to specify these, use the 'MTE' model.")
    args.warmup = True
    args.clip_grads = True
    args.BERT_layer_norm_placement = True
    args.no_non_linearity_in_attention_block = True

elif args.model in ['sm', 'softmax', 'mte', 'MTE']:
    params.update({'model': 'MTE'})
    experiment_name += 'MTE'

elif args.model in ['n', 'N', 'nap', 'norm', 'NAP', 'Normalized Attention Pooling']:
    params.update({'model': 'NAP'})
    experiment_name += 'NAP'

elif args.model in ['nn', 'no_norm', 'non', 'NON']:
    params.update({'model': 'NON'})
    experiment_name += 'NON'

elif args.model in ['s', 'sp', 'sum', 'sum_pooling']:
    params.update({'model': 'sum'})
    experiment_name += 'sum'

elif args.model in ['m', 'mp', 'max', 'max_pooling']:
    params.update({'model': 'max'})
    experiment_name += 'max'

elif args.model in ['weighted']:
    params.update({'model': 'w'})
    experiment_name += 'w'
    args.skip_validation = True  # cannot validate sequence length dependent model

else:
    raise NotImplementedError('Model not recognised')

if not params['model'] == 'BERT':
    if args.clip_grads:
        experiment_name += '-grads_clipped'
    if args.warmup:
        experiment_name += '-lr_warmup'
    if args.BERT_layer_norm_placement:
        experiment_name += '-BERT_ln'
    if args.no_non_linearity_in_attention_block:
        experiment_name += '-no_gelu'

RESULTS_FILE = experiment_name + '.pickle'


total_training_runs = args.training_runs * variations[0]['range'] * variations[1]['range']

try:
    results, start_from, run_order = pickle.load(open(RESULTS_FILE, 'rb'))
    args.reverse_run_order = run_order == 1
    args.random_run_order = run_order > 1
    print('Found existing results. Continuing on these (with run order specified there).')

except (OSError, IOError) as e:
    results = [[[[[] for _ in range(4 + 4 * NUM_CASES)]
                 for __ in range(variations[0]['range'])]
                for ___ in range(variations[1]['range'])]
               for ____ in range(params['epochs'])]
    start_from = 0
    pickle.dump([variations, params], open(experiment_name + '_params.pickle', 'wb'))
test_results = []

for run in range(start_from, total_training_runs):
    if args.random_run_order:
        indices = [np.random.randint(variations[0]['range']), np.random.randint(variations[1]['range'])]
    else:
        if args.reverse_run_order:
            # start with biggest models and iterate to smallest
            indices = [variations[0]['range'] - 1 - run % variations[0]['range'],
                       variations[1]['range'] - 1 - (run // variations[0]['range']) % variations[1]['range']]
        else:
            # start with smallest models and iterate to biggest
            indices = [run % variations[0]['range'], (run // variations[0]['range']) % variations[1]['range']]

    if xy and not np.all(np.equal(indices, xy)):
        continue

    for idx, variation in enumerate(variations):
        if variation['base']:
            params[variation['name']] = variation['base'] ** (indices[idx] + variation['min'])
        else:
            params[variation['name']] = indices[idx] + variation['min']

    if variations[0]['name'] == 'batch_size':
        # adjust steps_per_epoch to keep total data points seen constant
        params['steps_per_epoch'] = 100 * 32 // params['batch_size']
    elif variations[0]['name'] == 'vocabulary_size':
        # adjust steps_per_epoch to keep total data points seen per class approximately constant
        params['steps_per_epoch'] = 100 * params['vocabulary_size'] // 8

    print('-------------------')
    run_number = run // (total_training_runs // args.training_runs) + 1
    total_runs = args.training_runs if xy else total_training_runs
    print('Run ' + str(run_number) + ' of ' + str(total_runs))
    print(params)
    print(variations)
    print(experiment_name)
    print('-------------------')

    if args.task == 'ppi':
        history = create_and_run_model(**params)
    else:
        model = create_model(positional_embeddings=positional_embeddings,
                             no_non_linearity_in_attention_block=args.no_non_linearity_in_attention_block,
                             bert_layer_norm=args.BERT_layer_norm_placement,
                             last_layer=last_layer,
                             **params)
        model.summary()

        total_steps = params['steps_per_epoch'] * params['epochs']

        if args.warmup:
            learning_rate = WarmUpThenLinDecaySchedule(initial_learning_rate=params['learning_rate'],
                                                       warm_up_steps=total_steps // 10,
                                                       total_steps=total_steps)
        else:
            learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=params['learning_rate'],
                                                                          decay_steps=total_steps,
                                                                          end_learning_rate=0.0)

        optimizer = ClipAdam(learning_rate) if args.clip_grads else tf.keras.optimizers.Adam(learning_rate)

        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

        callbacks = []
        if args.task in ['cases', 'lg']:
            callbacks.append(CaseEvaluationCallback(results, indices, params['vocabulary_size'],
                                                    params['max_sequence_length'], args.task,
                                                    not args.skip_validation))
            train_generator = generator(params['vocabulary_size'], params['batch_size'], params['max_sequence_length'],
                                        params['case_bias'])
            # validate on sequences of half the length
            validation_generator = generator(params['vocabulary_size'], params['batch_size'],
                                             params['max_sequence_length'] // 2)
            if args.skip_validation:
                params['validation_steps'] = 0
        else:
            train_generator = generator(params['vocabulary_size'], params['batch_size'], params['max_sequence_length'])
            # validate on sequences of twice the length
            validation_generator = generator(params['vocabulary_size'], params['batch_size'],
                                             params['max_sequence_length'] * 2)

        history = model.fit(train_generator,
                            steps_per_epoch=params['steps_per_epoch'],
                            epochs=params['epochs'],
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=params['validation_steps'])

    for epoch in range(params['epochs']):
        try:
            results[epoch][indices[1]][indices[0]][0].append(history.history[metric_name][epoch])
            results[epoch][indices[1]][indices[0]][1].append(history.history['val_' + metric_name][epoch])
            results[epoch][indices[1]][indices[0]][2].append(history.history['loss'][epoch])
            results[epoch][indices[1]][indices[0]][3].append(history.history['val_loss'][epoch])
        except (IndexError, KeyError) as err:
            print(err)
    pickle.dump((results, run + 1, int(args.reverse_run_order) + 2 * int(args.random_run_order)),
                open(RESULTS_FILE, 'wb'))
    tf.keras.backend.clear_session()

