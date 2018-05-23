import sys
import statistics
import pickle

from eppsea_basicEA import ResultHolder, FitnessFunction, SelectionFunction

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

    formal_selection_names = {
        'eppsea_selection_function': 'Evolved Selection Function',
        'k_tournament': 'k-tournament',
        'k_tournament_2': 'k-tournament, k=2',
        'k_tournament_3': 'k-tournament, k=3',
        'k_tournament_5': 'k-tournament, k=5',
        'fitness_rank': 'Fitness Ranking',
        'fitness_proportional': 'Fitness Proportional',
        'random': 'Random',
        'truncation': 'Truncation'
    }

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
            if parent_selection_function == 'random':
                continue
            formal_selection_name = formal_selection_names[parent_selection_function]
            mu = final_result.get_eval_counts()
            fitness = final_result.get_average_best_fitness()

            errors = []
            for m in final_result.get_eval_counts():
                best_fitnesses = []
                for r in final_result.run_results:
                    if m in r['best_fitnesses']:
                        best_fitnesses.append(r['best_fitnesses'][m])
                    # if this run terminated early, use the final best fitness
                    elif all(m > rm for rm in r['best_fitnesses'].keys()):
                        best_fitnesses.append(r['final_best_fitness'])
                error = statistics.stdev(best_fitnesses)
                errors.append(error)

            plt.errorbar(mu, fitness, errors=None, label=formal_selection_name, errorevery=5)

        #plt.title('Average Best fitness, {0} function'.format(fitness_function_name))
        plt.xlabel('Evaluations')
        plt.ylabel('Best Fitness')
        plt.legend(loc=(1.02,0))
        plt.savefig('{0}/figure{1}.png'.format(results_directory, fitness_function_number), bbox_inches='tight')

        print('Plotting boxplot')
        final_best_fitnesses_list = []
        formal_selection_name_list = []
        fitness_function_name = results['eppsea_selection_function'].fitness_function_name
        if fitness_function_name in ['rosenbrock', 'rastrigin']:
            plt.yscale('symlog')
        for parent_selection_function, final_result in sorted(results.items()):
            if parent_selection_function == 'random':
                continue
            formal_selection_name = formal_selection_names[parent_selection_function]
            formal_selection_name_list.append(formal_selection_name)
            final_best_fitnesses = final_result.get_final_best_fitness_all_runs()
            final_best_fitnesses_list.append(final_best_fitnesses)
        plt.boxplot(final_best_fitnesses_list, labels=formal_selection_name_list)

        plt.xlabel('Evaluations')
        plt.xticks(rotation=90)
        plt.ylabel('Final Best Fitness')
        legend = plt.legend([])
        legend.remove()
        plt.savefig('{0}/boxplot{1}.png'.format(results_directory, fitness_function_number), bbox_inches='tight')


        print('Doing t-tests')
        eppsea_results = results['eppsea_selection_function']
        sample1 = eppsea_results.get_final_best_fitness_all_runs()
        sample1_mean = statistics.mean(sample1)
        print('Mean performance of EPPSEA function on fitness function {0}: {1}'.format(fitness_function_number, sample1_mean))
        for parent_selection_function, final_result in results.items():
            if parent_selection_function != 'eppsea_selection_function':
                sample2 = final_result.get_final_best_fitness_all_runs()
                sample2_mean = statistics.mean(sample2)
                mean_difference = sample1_mean - sample2_mean

                t, p = scipy.stats.ttest_rel(sample1, sample2)

                print('Mean performance of {0} selection: {1} | Difference versus EPPSEA: {2} | p-value: {3} '.format(parent_selection_function, sample2_mean, mean_difference, p))


if __name__ == '__main__':
    main()