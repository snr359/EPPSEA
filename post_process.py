import sys
import statistics
import pickle

from eppsea_basicEA import ResultHolder, FitnessFunction

import scipy.stats

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
    final_results_path = sys.argv[1]
    results_directory = sys.argv[2]

    print('Loading final results for postprocessing')
    with open(final_results_path, 'rb') as final_results_file:
        all_final_results = pickle.load(final_results_file)

    for fitness_function_number, fitness_function_results in all_final_results.items():
        plt.clf()
        print('Analyzing results for fitness function {0}'.format(fitness_function_number))
        results = dict()
        for r in fitness_function_results:
            results[r.selection_function_name] = r

        print('Plotting figure')
        fitness_function_name = results['eppsea_selection_function'].fitness_function_name
        if fitness_function_name in ['rosenbrock', 'rastrigin']:
            plt.yscale('symlog')
        for parent_selection_function, final_result in sorted(results.items()):
            mu = final_result.get_eval_counts()
            fitness = final_result.get_average_best_fitness()
            plt.plot(mu, fitness, label=parent_selection_function)

        plt.title('Average Best fitness, {0} function'.format(fitness_function_name))
        plt.xlabel('Evaluations')
        plt.ylabel('Best Fitness')
        plt.legend()
        plt.savefig('{0}/figure{1}.png'.format(results_directory, fitness_function_number))

        print('Doing t-tests')
        eppsea_results = results['eppsea_selection_function']
        for parent_selection_function, final_result in results.items():
            if parent_selection_function != 'eppsea_selection_function':
                sample1 = eppsea_results.get_final_best_fitness_all_runs()
                sample2 = final_result.get_final_best_fitness_all_runs()
                sample1_mean = statistics.mean(sample1)
                sample2_mean = statistics.mean(sample2)
                mean_difference = sample1_mean - sample2_mean

                t, p = scipy.stats.ttest_rel(sample1, sample2)

                print('Mean difference in final best fitness on fitness function {0} compared to {1} selection: {2}. p-value: {3}'.format(fitness_function_number, parent_selection_function, mean_difference, p))


if __name__ == '__main__':
    main()