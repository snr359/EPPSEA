# this script finds the best k value for k-tournament selection given a configuration for the eppsea_basicEA script, and a number of processes to run
import sys
import configparser
import multiprocessing
import statistics
import math

import eppsea_basicEA

def test_k_tournament(basic_ea, k):
    # tests a value for k tournament selection and returns the average best fitness of the EA
    selection_functions = [('k_tournament', None)]
    eas = basic_ea.get_eas(selection_functions, True)
    for ea in eas:
        ea.tournament_k = k
    results = basic_ea.run_eas(eas)[0]
    average_best = results.get_average_final_best_fitness()
    return average_best

def find_optimal_k(basic_ea, num_interval_points=None):
    min_k = 1
    max_k = basic_ea.config.getint('EA', 'population size')

    # if num_processes is None or invalid, approximate a value for a two-iteration approach
    if num_interval_points is None or num_interval_points < 4:
        num_interval_points = max(4, math.ceil(math.sqrt(max_k)))

    # save old k-tournament value so it can be reassigned later
    old_tournament_k = basic_ea.tournament_k

    # sample for an interval for the optimum value by sectioning (assuming the optimal k-value is a local optimum among all possible k values)
    known_values = dict()

    while (max_k - min_k)+1 > num_interval_points:
        # determine evenly spaced k values
        stepsize = (max_k - min_k) / (num_interval_points - 1)
        k_values = []
        for i in range(num_interval_points - 1):
            k_values.append(round(min_k + i*stepsize))
        k_values.append(max_k)

        # test the k values
        print('Testing with k-values: {0}'.format(k_values))
        results = []
        for k in k_values:
            if k in known_values:
                result = known_values[k]
            else:
                result = test_k_tournament(basic_ea, k)

            results.append(result)
            known_values[k] = result

        # if the first value is the maximum, set min and max to the first and second values
        if results[0] == max(results):
            min_k = k_values[0]
            max_k = k_values[1]

        # if the last value is the maximum, set min and max to the second to last and last values
        elif results[-1] == max(results):
            min_k = k_values[-2]
            max_k = k_values[-1]

        # otherwise, find the highest k value
        else:
            top_index = results.index(max(results))

            # set the new min and max values to the values around the peak k value
            min_k = k_values[top_index-1]
            max_k = k_values[top_index+1]

    # once there is a small interval for optimal k values, test every value in that interval
    k_values = list(range(min_k, max_k+1))
    print('Testing with k-values: {0}'.format(k_values))
    results = []
    for k in k_values:
        if k in known_values:
            result = known_values[k]
        else:
            result = test_k_tournament(basic_ea, k)

        results.append(result)
        known_values[k] = result

    # calculate the best k values
    top = results.index(max(results))

    # re-set old k-tournament value before returning
    basic_ea.tournament_k = old_tournament_k

    return k_values[top]

if __name__ == '__main__':

    if len(sys.argv) < 3:
        num_interval_points = None
    else:
        num_interval_points = int(sys.argv[2])

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    # create a basic EA object to do the evaluations with
    basic_ea = eppsea_basicEA.EppseaBasicEA(config)

    optimal_k = find_optimal_k(basic_ea, num_interval_points)

    print('Best k tournament value: {0}'.format(optimal_k))
