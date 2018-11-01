import statistics
import sys
import pickle

import numpy as np

import scipy.stats

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def get_eval_counts(results):
    all_eval_counts = []
    for r in results:
        all_eval_counts.extend(r['eval_counts'])
    return sorted(list(set(all_eval_counts)))

def main(final_output_directory, results_file_paths):
    # first, load all results
    results = []
    for fp in results_file_paths:
        with open(fp, 'rb') as file:
            results.extend(pickle.load(file))
            
    # get the fitness function names and ids
    fitness_function_ids = []
    fitness_function_display_names = dict()
    
    for r in results:
        if r['fitness_function_id'] not in fitness_function_ids:
            fitness_function_ids.append(r['fitness_function_id'])
            fitness_function_display_names[r['fitness_function_id']] = r['fitness_function_display_name']

    # get the selection function names and ids
    selection_function_ids = []
    selection_function_display_names = dict()

    for r in results:
        if r['selection_function_id'] not in selection_function_ids:
            selection_function_ids.append(r['selection_function_id'])
            selection_function_display_names[r['selection_function_id']] = r['selection_function_display_name']
    
    np.seterr(all='warn')
    # log string forms of eppsea-based selection functions
    printed_selection_functions = []
    for r in results:
        if r['selection_function_eppsea_string'] not in printed_selection_functions:
            print('String form of {0}: {1}'.format(r['selection_function_display_name'], r['selection_function_eppsea_string']))
            printed_selection_functions.append(r['selection_function_eppsea_string'])
    # Analyze results for each fitness function
    for fitness_function_id in fitness_function_ids:
        plt.clf()
        print('Analyzing results for fitness function with id {0} ---------------------------------\n'.format(fitness_function_id))
        print('Plotting figure\n')
        # Get the name of the fitness function from one of the result files
        fitness_function_name = fitness_function_display_names[fitness_function_id]

        # filter out the results for this fitness function
        fitness_function_results = list(r for r in results if r['fitness_function_id'] == fitness_function_id)

        # Set the plot to use Log Scale
        plt.yscale('symlog')

        # Plot results for each selection function
        for selection_function_id in selection_function_ids:
            selection_function_results = list(r for r in fitness_function_results if r['selection_function_id'] == selection_function_id)
            selection_function_name = selection_function_display_names[selection_function_id]
            mu = get_eval_counts(selection_function_results)
            average_best_fitnesses = []
            for m in mu:
                average_best_fitnesses.append(statistics.mean(
                    r['best_fitnesses'][m] for r in selection_function_results if m in r['best_fitnesses']))

            plt.plot(mu, average_best_fitnesses, label=selection_function_name)

        plt.xlabel('Evaluations')
        plt.ylabel('Best Fitness')
        plt.legend(loc=(1.02, 0))
        plt.savefig('{0}/figure_{1}.png'.format(final_output_directory, fitness_function_id),
                    bbox_inches='tight')

        print('Plotting boxplot\n')
        final_best_fitnesses_list = []
        selection_name_list = []

        # Set the plot to use Log Scale
        plt.yscale('symlog')

        for selection_function_id in selection_function_ids:
            selection_function_results = list(r for r in fitness_function_results if r['selection_function_id'] == selection_function_id)
            selection_function_name = selection_function_display_names[selection_function_id]
            selection_name_list.append(selection_function_name)
            final_best_fitnesses = list(r['final_best_fitness'] for r in selection_function_results)
            final_best_fitnesses_list.append(final_best_fitnesses)
        plt.boxplot(final_best_fitnesses_list, labels=selection_name_list)

        plt.xlabel('Evaluations')
        plt.xticks(rotation=90)
        plt.ylabel('Final Best Fitness')
        legend = plt.legend([])
        legend.remove()
        plt.savefig('{0}/boxplot_{1}.png'.format(final_output_directory, fitness_function_id),
                    bbox_inches='tight')

        print('Doing t-tests\n')

        tested_pairs = []

        for selection_function_id1 in selection_function_ids:
            selection_function_results1 = list(r for r in fitness_function_results if r['selection_function_id'] == selection_function_id1)
            selection_function_target_results1 = list(r for r in selection_function_results1 if r['termination_reason'] == 'target_fitness_hit')
            selection_function_name1 = selection_function_display_names[selection_function_id1]
            final_best_fitnesses1 = list(r['final_best_fitness'] for r in selection_function_results1)
            # round means to 5 decimal places for cleaner display
            average_final_best_fitness1 = round(statistics.mean(final_best_fitnesses1), 5)
            target_hit_percentage1 = len(selection_function_target_results1) / len(selection_function_results1)
            print('Mean performance of {0}: {1}, reaching target fitness in {2}% of runs\n'.format(selection_function_name1,average_final_best_fitness1, target_hit_percentage1))
            # perform a t test with all the other results that this selection has not yet been tested against
            for selection_function_id2 in selection_function_ids:
                if selection_function_id2 != selection_function_id1 and (selection_function_id1, selection_function_id2) not in tested_pairs and (selection_function_id2, selection_function_id1) not in tested_pairs:
                    selection_function_results2 = list(r for r in fitness_function_results if r['selection_function_id'] == selection_function_id2)
                    selection_function_target_results2 = list(r for r in selection_function_results2 if r['termination_reason'] == 'target_fitness_hit')
                    selection_function_name2 = selection_function_display_names[selection_function_id2]
                    final_best_fitnesses2 = list(r['final_best_fitness'] for r in selection_function_results2)
                    # round means to 5 decimal places for cleaner display
                    average_final_best_fitness2 = round(statistics.mean(final_best_fitnesses2), 5)
                    target_hit_percentage2 = len(selection_function_target_results2) / len(selection_function_results2)
                    print('Mean performance of {0}: {1}, reaching target fitness in {2}% of runs\n'.format(selection_function_name2, average_final_best_fitness2,target_hit_percentage2))

                    _, p_fitness = scipy.stats.ttest_rel(final_best_fitnesses1, final_best_fitnesses2)
                    mean_difference_fitness = round(average_final_best_fitness1 - average_final_best_fitness2, 5)

                    if p_fitness < 0.05:
                        if mean_difference_fitness > 0:
                            print('\t\t{0} performed {1} better | p-value: {2}\n'.format(selection_function_name1,mean_difference_fitness,p_fitness))
                        else:
                            print('\t\t{0} performed {1} worse | p-value: {2}\n'.format(selection_function_name1,mean_difference_fitness,p_fitness))
                    else:
                        print('\t\t{0} performance difference is insignificant | p-value: {1}\n'.format(selection_function_name1, p_fitness))

                    final_target_evals1 = list(max(r['eval_counts']) for r in selection_function_target_results1)
                    final_target_evals2 = list(max(r['eval_counts']) for r in selection_function_target_results2)

                    if len(final_target_evals1) > 0 and len(final_target_evals2) > 0:

                        _, p_evals = scipy.stats.ttest_rel(final_target_evals1, final_target_evals2)
                        mean_difference_evals = round(statistics.mean(final_target_evals1) - statistics.mean(final_target_evals2), 5)

                        if p_evals < 0.05:
                            if mean_difference_evals < 0:
                                print('\t\t{0} used {1} fewer evals to hit target fitness | p-value: {2}\n'.format(selection_function_name1,mean_difference_evals,p_evals))
                            else:
                                print('\t\t{0} used {1} more evals to hit target fitness | p-value: {2}\n'.format(selection_function_name1,mean_difference_evals,p_evals))
                        else:
                            print('\t\t{0} difference in eval count to target fitness is insignificant | p-value: {1}\n'.format(selection_function_name1, p_evals))

                    tested_pairs.append((selection_function_id1, selection_function_id2))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide output directory and results file(s)')
        exit(1)
    final_output_directory = sys.argv[1]
    results_file_paths = sys.argv[2:]
    main(final_output_directory, results_file_paths)