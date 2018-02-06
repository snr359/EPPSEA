# this script finds the best k value for k-tournament selection given a configuration for the eppsea_basicEA script
import sys
import configparser
import multiprocessing
import statistics

import eppsea_basicEA

def test_k_tournament(basic_ea, k):
    # tests a value for k tournament selection and returns the average best fitness of the EA
    results = []
    basic_ea.tournament_k = k
    for _ in range(basic_ea.runs):
        results.append(basic_ea.one_run('k_tournament'))

    average_best = statistics.mean(r['best_fitness'] for r in results)

    return average_best

if __name__ == '__main__':

    num_processes = 8

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    # create a basic EA object to do the evaluations with
    basic_ea = eppsea_basicEA.basicEA(config)

    # sample for an interval for the optimum value by sectioning (assuming the optimal k-value is a local optimum among all possible k values)
    min_k = 1
    max_k = basic_ea.mu

    while (max_k - min_k)+1 > num_processes:
        # determine evenly spaced k values
        stepsize = (max_k - min_k) / (num_processes - 1)
        k_values = []
        for i in range(num_processes):
            k_values.append(round(min_k + i*stepsize))

        # test the k values
        print('Testing with k-values: {0}'.format(k_values))
        pool = multiprocessing.Pool(processes=num_processes)
        params = list((basic_ea, k) for k in k_values)
        results = pool.starmap(test_k_tournament, params)
        pool.close()

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
    pool = multiprocessing.Pool(processes=num_processes)
    params = list((basic_ea, k) for k in k_values)
    results = pool.starmap(test_k_tournament, params)
    pool.close()

    # calculate the best k values
    top = results.index(max(results))
    print('Best k tournament value: {0}'.format(k_values[top]))
