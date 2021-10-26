import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pickle
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cases', choices=['cases', 'mode', 'lg', 'ppi', 'rl'],
                    help="The task to load results from. Can be either 'cases' for the case distinction/pin-pointing "
                         "task or 'mode' for the mode finding task. Defaults to 'cases'.")
parser.add_argument('-f', '--first_token_output', action='store_true',
                    help='Flag to switch from results with the output taken across all tokens to results with the '
                         'output taken from the first token in the case distinction task.')
parser.add_argument('-t', '--training_runs', type=int, default=5,
                    help='Expected number of results per hyper parameter combination. Defaults (as in train.py) to 5.')
parser.add_argument('--variation', type=str, default='model_dimension',
                    help="Option to specify the hyperparamerer sweep from which to load the results. Can be "
                         "abbreviated, e.g., 'b' for 'batch_size'. Defaults to 'model_dimension'.")
parser.add_argument('--model', type=str, default='all',
                    help="The architecture model from which the results should be loaded. Defaults to 'all', "
                         "i.e., all results that are found for the specified task and variation.")
parser.add_argument('-m', '--min_mean_max', action='store_true',
                    help='Flag to load and display min-, mean- and max-accuracies as RGB pixel values.')
parser.add_argument('-c', '--cases', action='store_true',
                    help="Flag to load and display argmin-, first- and argmax-mean-case-accuracies as RGB pixel values."
                         " This option is only available if task is set to 'cases'.")
parser.add_argument('-cc', '--all_case_accuracies', action='store_true',
                    help="Flag to load and display argmin-, first- and argmax- min-, mean- and max-case-accuracies. "
                         "This option is only available if task is set to 'cases'. WARNING: This option tends to open "
                         "a lot of result plots, as a min-/mean-/max-accuracy plot is generated per case!")
parser.add_argument('-v', '--validation', action='store_true',
                    help="Flag to additionally display validation results.")
parser.add_argument('-r', '--robustness', action='store_true',
                    help="Flag to additionally display robustness curves of the results.")
parser.add_argument('-a', '--animate', action='store_true',
                    help="Flag to animate plot evolution over the course of training. WARNING: This option will open "
                         "an animation per model! Consider specifying the model of interest with the '--model' option.")
parser.add_argument('-l', '--learning_curve', action='store_true',
                    help="Flag to display learning curves instead of RGB plots.")
parser.add_argument('-p', '--print_numbers', action='store_true',
                    help="Prints numeric values of the result.")
parser.add_argument('-pm', '--print_max_numbers', action='store_true',
                    help="Prints only max numeric values of the result.")
parser.add_argument('--prefix', type=str, default='paper_results/',
                    help="Prefix to specify the location from where the results should be loaded. Defaults to the "
                         "'paper_results' folder.")
args = parser.parse_args()

if not args.min_mean_max and not args.cases and not args.all_case_accuracies:
    raise NotImplementedError('Please specify which results you would like to display, '
                              'e.g., add -m for min/mean/max accuracy results')
prefix = args.prefix + args.task + '/'

if args.first_token_output:
    prefix += 'first_token_output/'

if args.variation in ['d', 'dim', 'model_dimension']:
    prefix += 'var_dim/'

elif args.variation in ['l', 'L', 'layers']:
    prefix += 'var_layers/'

elif args.variation in ['i', 'I', 'init', 'initialization', 'initial_std']:
    prefix += 'var_init/'

elif args.variation in ['bias', 'case_bias']:
    prefix += 'var_bias/'

elif args.variation in ['h', 'H', 'm', 'M', 'heads']:
    prefix += 'var_heads/'

elif args.variation in ['s', 'S', 'seq', 'len', 'max_sequence_length']:
    prefix += 'var_seq/'

elif args.variation in ['b', 'bs', 'batch', 'batch_size']:
    prefix += 'var_batch/'

elif args.variation in ['v', 'vs', 'vocab', 'vocabulary_size']:
    prefix += 'var_vocab/'

else:
    raise NotImplementedError('Variation not recognised')

model_names = [file.replace('.pickle', '')
               for file in listdir(prefix) if isfile(join(prefix, file)) and 'params' not in file]

if args.model != 'all':
    model_names = [model_name for model_name in model_names if model_name in args.model]

print('\n----> Showing results from', prefix, '<----')

if not model_names:
    raise FileNotFoundError('No results found for the specified task/variation/model combination.')

result_array = []
for model in model_names:
    print('\nLoading results from model ', model)
    variations, params = pickle.load(open(prefix + model + '_params.pickle', 'rb'))
    print('Results correspond to variations\n', variations, '\nwith default parameters\n', params)
    results, completed_runs, run_order = pickle.load(open(prefix + model + '.pickle', 'rb'))
    if completed_runs != args.training_runs * variations[0]['range'] * variations[1]['range']:
        print('WARNING: Results are incomplete')
    elif run_order > 1:
        print('WARNING: Run order was random. '
              'The number of random seeds per hyper-parameter combination is not consistent.')
    result_array.append(np.asarray(results).tolist())

print('\n')
result_array = np.asarray(result_array)
if len(result_array.shape) < 5:
    raise AssertionError("Results don't match")
if len(result_array.shape) < 6:
    results_flat = np.reshape(result_array, (-1,))
    min_run_number = np.min(list(map(len, results_flat)))
    print('WARNING: Some combinations have less results than others.')
    print('All combinations have at least', min_run_number, 'result(s).') if min_run_number else \
        print('And some combinations have no results.')

    def maybe_aggregate(aggregation_function):
        """Wrapper that sets results to 0 where there are no results yet."""
        def aggregation_wrapper(result_list, enumerator_list, *args, **kwargs):
            if result_list:
                if enumerator_list == 1:
                    return aggregation_function(result_list, *args, **kwargs)
                ratios = np.asarray(result_list, dtype=np.float) / np.asarray(enumerator_list, dtype=np.float)
                return aggregation_function(ratios, *args, **kwargs)
            else:
                return 0.0
        return aggregation_wrapper

    def get_min_mean_max(results, enumerator=None):
        flat_results = np.reshape(results, (-1,))
        if enumerator is None:
            flat_enumerator = np.ones_like(flat_results)
        else:
            flat_enumerator = np.reshape(enumerator, (-1,))
        min_mean_max_results = []
        for func in [np.min, np.mean, np.max]:
            min_mean_max_results.append(np.reshape(list(map(maybe_aggregate(func), flat_results, flat_enumerator)),
                                                   results.shape[:4]))
        return np.stack(min_mean_max_results, axis=-1)
else:
    min_run_number = None

    def get_min_mean_max(results, enumerator=None):
        if enumerator is not None:
            results /= enumerator
        return np.stack([np.min(results, axis=-1), np.mean(results, axis=-1), np.max(results, axis=-1)], axis=-1)

robustness = []
for model_id, model in enumerate(model_names):
    if min_run_number is not None:
        flat_results = np.reshape(result_array, (-1,))
        tmp_results = np.asarray(list(map(lambda result: result[:min_run_number],
                                          flat_results))).reshape(result_array.shape + (min_run_number,))
    else:
        tmp_results = result_array
    model_validation_results = tmp_results[model_id, :, :, :, 1]
    hyperpar_index = np.unravel_index(np.argmax(np.mean(np.max(model_validation_results, axis=0), axis=-1)),
                                      (variations[1]['range'], variations[0]['range']))
    epoch_indices = np.argmax(model_validation_results[:, hyperpar_index[0], hyperpar_index[1]], axis=0)
    test_idx = 3 if args.task == 'ppi' else 1  # only ppi has actual test results, otherwise take validation results
    test_results = []
    for run, epoch_idx in enumerate(epoch_indices):
        test_results.append(tmp_results[model_id, epoch_idx, hyperpar_index[0], hyperpar_index[1], test_idx, run])

    sorted_results = np.sort(np.mean(np.max(model_validation_results, axis=0), axis=-1).reshape((-1,)))[::-1]
    robustness.append(sorted_results)

    print('------')
    print(model)
    print('Best results at: ', reversed(hyperpar_index))
    print(test_results)
    print(np.mean(test_results), '+/-', np.std(test_results))

if args.robustness:
    plt.title('Robustness of the models')
    plt.xlabel('Hyperparamter combinations')
    plt.ylabel('Accuracy')
    for model_robustness in robustness:
        plt.plot(np.transpose(model_robustness))
    plt.legend(model_names)
    plt.show()

results_to_display = {}


def load_case_results(validation=False):
    if args.task not in ['cases', 'lg']:
        raise NotImplementedError('The `cases` task and `lg` task have a case distinction and case results.')
    case_result_indices = [4, 6, 9] if args.task == 'cases' else [4, 5, 5]
    num_cases = 7 if args.task == 'cases' else 2
    name_addition = ''
    if validation:
        case_result_indices = [18, 20, 23] if args.task == 'cases' else [8, 9, 9]
        name_addition = 'Validation '
    case_results = []
    for case, case_result_idx in zip(['argmin', 'first', 'argmax'], case_result_indices):
        case_result = get_min_mean_max(result_array[:, :, :, :, case_result_idx].copy(),
                                       result_array[:, :, :, :, case_result_idx + num_cases].copy())
        if args.all_case_accuracies:
            results_to_display.update({'Case ' + case + ' Min Mean Max ' + name_addition: case_result})
        case_results.append(case_result)
    all_case_results = np.stack(case_results, axis=-1)
    results_to_display.update({'Mean Case Accuracies ' + name_addition: all_case_results[:, :, :, :, 1]})
    results_to_display.update({'All Case Accuracies ' + name_addition: all_case_results})


if args.min_mean_max:
    results_to_display.update({'Min Mean Max ': get_min_mean_max(result_array[:, :, :, :, 0])})
    if args.validation:
        results_to_display.update({'Min Mean Max Validation ': get_min_mean_max(result_array[:, :, :, :, 1])})
if args.cases or args.all_case_accuracies:
    load_case_results()
    if args.validation:
        load_case_results(True)


xrange = range(variations[0]['range'])
yrange = range(variations[1]['range'])


def display_images(images, name, only_best=True):
    if 'All' in name:
        return

    def plot_epoch_images(result_image):
        for model_id, model in enumerate(model_names):
            plt.subplot(round(float(len(model_names)) / 2.0 + 0.1), 2, model_id + 1)
            plt.imshow(result_image[model_id], interpolation=None)
            plt.title(model)
            if args.print_numbers or args.print_max_numbers:
                print('\n', model + name)
                max_values = [-np.inf, -np.inf, -np.inf]
                max_values_idx = [0, 0]
                for x_idx, row in enumerate(result_image[model_id]):
                    for y_idx, value in enumerate(row):
                        if value[1] > max_values[1]:
                            max_values = value
                            max_values_idx = [y_idx, x_idx]
                        if args.print_numbers:
                            print(y_idx, x_idx, str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]))
                print('Max values: ', max_values, 'at', max_values_idx)
            plt.xlabel(variations[0]['name'])
            if variations[0]['base']:
                plt.xticks(xrange, [round(variations[0]['base'] ** (exp + variations[0]['min']),
                                          variations[0]['range']) for exp in xrange])
            else:
                plt.xticks(xrange, [x + variations[0]['min'] for x in xrange])
            plt.ylabel(variations[1]['name'])
            if variations[1]['base']:
                plt.yticks(yrange, [round(variations[1]['base'] ** (exp + variations[1]['min']),
                                          variations[1]['range']) for exp in yrange])
            else:
                plt.yticks(yrange, [y + variations[1]['min'] for y in yrange])
        plt.tight_layout()
    if only_best:
        result_image = np.max(images, axis=1)
        plt.figure(name + 'Best Epoch')
        plot_epoch_images(result_image)
    else:
        for epoch, result_image in enumerate(images):
            plt.figure(name + 'Epoch ' + str(epoch + 1))
            plot_epoch_images(result_image)


def display_animation(images, name):
    if 'All' in name:
        return

    for model_id, model in enumerate(model_names):
        fig = plt.figure(name + model)
        frames = []
        for epoch, result_image in enumerate(images[model_id]):
            frames.append([plt.imshow(result_image, animated=True)])
            plt.title(model)
            plt.xlabel(variations[0]['name'])
            if variations[0]['base']:
                plt.xticks(xrange, [round(variations[0]['base'] ** (exp + variations[0]['min']),
                                          variations[0]['range']) for exp in xrange])
            else:
                plt.xticks(xrange, [x + variations[0]['min'] for x in xrange])
            plt.ylabel(variations[1]['name'])
            if variations[1]['base']:
                plt.yticks(yrange, [round(variations[1]['base'] ** (exp + variations[1]['min']),
                                          variations[1]['range']) for exp in yrange])
            else:
                plt.yticks(yrange, [y + variations[1]['min'] for y in yrange])
        print("Creating your animation...")
        animation = ani.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
        animation.save('animations/' + model + '.gif', writer='imagemagick', fps=6)
        print("Your animation is saved to: animations/" + model + '.gif')
        plt.close()


def show_learning_curves(images, name):
    if 'Mean Case' in name:
        return

    min_mean_max = 'Min Mean Max' in name
    if min_mean_max:
        plt.figure(name)
    clip_min = 0.0
    clip_max = 1.0
    for model_id, model in enumerate(model_names):
        if not min_mean_max:
            plt.figure(name + model)
        for x_idx in xrange:
            for y_idx in yrange:
                plt.subplot(variations[1]['range'], variations[0]['range'], x_idx + y_idx * variations[0]['range'] + 1)
                if min_mean_max:
                    plt.plot(images[model_id, :, y_idx, x_idx, 1])
                    epochs = range(len(images[model_id, :, y_idx, x_idx, 0]))
                    plt.fill_between(epochs, images[model_id, :, y_idx, x_idx, 0], images[model_id, :, y_idx, x_idx, 2],
                                     alpha=0.3)
                else:
                    epochs = range(len(images[model_id, :, y_idx, x_idx, 0, 0]))
                    for case in range(3):
                        plt.plot(np.mean(images[model_id, :, y_idx, x_idx, :, case], axis=-1))
                        plt.fill_between(epochs,
                                         np.min(images[model_id, :, y_idx, x_idx, :, case], axis=-1),
                                         np.max(images[model_id, :, y_idx, x_idx, :, case], axis=-1),
                                         alpha=0.3)
                plt.ylim(clip_min, clip_max)
                plt.xticks([], [])
                plt.yticks([], [])
        if min_mean_max:
            plt.legend(model_names)
        else:
            plt.legend(['argmin', 'first', 'argmax'])


if args.animate:
    display_function = display_animation
elif args.learning_curve:
    display_function = show_learning_curves
else:
    display_function = display_images

for key in results_to_display.keys():
    display_function(results_to_display[key], key)
plt.show()
