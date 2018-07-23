# this post-processing script loads results from EA runs and performs post-processing on them.
# the first command line argument is the output directory. The remaining command line arguments are paths to JSON dumps
# of dictionaries encoded with the following mappings:
# 'Name': A string, the identifying name of the EA, including selection functions used
# 'T Test': A boolean, signifying whether this result should be compared to all the others with a T-test
# 'Log Scale': A boolean, signifying whether this data should be plotted in log-scale
# 'Fitness Functions': A dictionary, mapping numbers (each representing a fitness function) to dictionaries with the
# following mappings:
#   'Name': The name of the fitness function
#   'Runs': A list of dictionaries mapping eval counts to the best fitness achieved at that evaluation count for each run

import sys
import statistics
import json

import scipy.stats

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def fix_json_keys(x):
    # if x is a dictionary, return a new version of x where keys are converted to ints, if possible
    if isinstance(x, dict):
        # create a new mapping
        new_dict = dict()
        for k,v in x.items():
            # if the key is convertible to an integer, convert it
            try:
                int_k = int(k)
                new_dict[int_k] = v
            # otherwise, just re-record the key-value pair
            except ValueError:
                new_dict[k] = v
        return new_dict
    else:
        return x

def get_eval_counts(run_results):
    # Takes a list of dictionaries mapping evals to best fitnesses for each run, and returns a list of eval counts with
    # a recorded best fitness in at least one of the runs
    eval_counts = set()
    for r in run_results:
        eval_counts = eval_counts.union(r.keys())
    eval_counts = sorted(eval_counts)
    return eval_counts

def get_average_best_fitnesses(run_results):
    # Takes a list of dictionaries mapping evals to best fitnesses for each run, and returns the average best fitnesses
    # for each eval count. If a run does not have a best fitness for a given eval count, the best fitness for the next
    # highest eval count is used instead. All runs must have the same first eval count
    average_best_fitnesses = dict()
    # Get the eval counts
    eval_counts = get_eval_counts(run_results)
    # Get the average best fitness at each eval count
    for e in eval_counts:
        best_fitnesses = []
        for r in run_results:
            if e in r:
                best_fitnesses.append(r[e])
            # If this run does not have a fitness value for this eval count, use the fitness value from the previous eval count
            else:
                previous_e = max(re for re in r.keys() if re < e)
                best_fitnesses.append(r[previous_e])

        average_best_fitness = statistics.mean(best_fitnesses)
        average_best_fitnesses[e] = average_best_fitness

    return average_best_fitnesses

def get_final_best_fitnesses(run_results):
    # takes a list of dictionaries mapping eval counts to fitnesses for each run, and returns a list of the final best
    # fitnesses for all the runs
    final_best_fitnesses = []
    for r in run_results:
        final_eval_count = max(r.keys())
        final_best_fitnesses.append(r[final_eval_count])

    return final_best_fitnesses

def main(final_output_directory, results_file_paths):
    # load the result dictionaries
    print('Loading final results for postprocessing')
    results = []
    for results_file_path in results_file_paths:
        with open(results_file_path) as results_file:
            result = json.load(results_file, object_hook=fix_json_keys)
            results.append(result)

    # Get the fitness function numbers
    fitness_function_numbers = set()
    for r in results:
        fitness_function_numbers = fitness_function_numbers.union(r['Fitness Functions'].keys())
    fitness_function_numbers = sorted(fitness_function_numbers)


    # Analyze results for each fitness function
    for fitness_function_number in fitness_function_numbers:
        plt.clf()
        print('Analyzing results for fitness function {0} ---------------------------------'.format(fitness_function_number))
        print('Plotting figure')
        # Get the name of the fitness function from one of the result files
        fitness_function_name = results[0]['Fitness Functions'][fitness_function_number]['Name']
        # Set the plot to use Log Scale if any of the result files require it
        if any(r['Log Scale'] for r in results):
            plt.yscale('symlog')

        # Plot each result
        for r in results:
            run_results = r['Fitness Functions'][fitness_function_number]['Runs']
            mu = get_eval_counts(run_results)
            average_best_fitnesses = get_average_best_fitnesses(run_results).values()

            plt.plot(mu, average_best_fitnesses, label=r['Name'])

        plt.xlabel('Evaluations')
        plt.ylabel('Best Fitness')
        plt.legend(loc=(1.02,0))
        plt.savefig('{0}/figure{1}.png'.format(final_output_directory, fitness_function_number), bbox_inches='tight')

        print('Plotting boxplot')
        final_best_fitnesses_list = []
        selection_name_list = []
        if any(r['Log Scale'] for r in results):
            plt.yscale('symlog')
        for r in results:
            run_results = r['Fitness Functions'][fitness_function_number]['Runs']
            selection_name_list.append(r['Name'])
            final_best_fitnesses = get_final_best_fitnesses(run_results)
            final_best_fitnesses_list.append(final_best_fitnesses)
        plt.boxplot(final_best_fitnesses_list, labels=selection_name_list)

        plt.xlabel('Evaluations')
        plt.xticks(rotation=90)
        plt.ylabel('Final Best Fitness')
        legend = plt.legend([])
        legend.remove()
        plt.savefig('{0}/boxplot{1}.png'.format(final_output_directory, fitness_function_number), bbox_inches='tight')


        print('Doing t-tests')
        for r1 in results:
            if r1['T Test']:
                r1_run_results = r1['Fitness Functions'][fitness_function_number]['Runs']
                sample1 = get_final_best_fitnesses(r1_run_results)
                # round mean to 5 decimal places for cleaner display
                sample1_mean = round(statistics.mean(sample1), 5)
                print('Mean performance of {0}: {1}'.format(r1['Name'], sample1_mean))
                for r2 in results:
                    if r2 is not r1:
                        r2_run_results = r2['Fitness Functions'][fitness_function_number]['Runs']
                        sample2 = get_final_best_fitnesses(r2_run_results)
                        sample2_mean = round(statistics.mean(sample2), 5)
                        t, p = scipy.stats.ttest_rel(sample1, sample2)
                        mean_difference = round(sample1_mean - sample2_mean, 5)

                        if p < 0.05:
                            if mean_difference > 0:
                                print('Mean performance of {0}: {1} | {2} performed {3} better | p-value: {4} '.format(r2['Name'], sample2_mean, r1['Name'], mean_difference, p))
                            else:
                                print('Mean performance of {0}: {1} | {2} performed {3} worse | p-value: {4} '.format(r2['Name'], sample2_mean, r1['Name'], mean_difference, p))
                        else:
                            print('Mean performance of {0}: {1} | {2} performance difference is insignificant | p-value: {4} '.format(r2['Name'], sample2_mean, r1['Name'], mean_difference, p))

if __name__ == '__main__':
    final_output_directory = sys.argv[1]
    results_file_paths = sys.argv[2:]
    main(final_output_directory, results_file_paths)