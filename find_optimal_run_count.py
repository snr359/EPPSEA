# finds out the optimal number of runs for a basicEA to achieve the precision passed in in sys.argv[2], using the config in sys.argv[1]

import multiprocessing
import sys
import math
import configparser
import statistics

import eppsea_basicEA

def main():
    config_path = sys.argv[1]
    precision = float(sys.argv[2])

    config = configparser.ConfigParser()
    config.read(config_path)

    basic_ea = eppsea_basicEA.basicEA(config)

    for parent_selection_function in ['truncation', 'fitness_proportional', 'fitness_rank', 'k_tournament']:
        done = False

        additions_within_precision = 0
        num_runs = 0
        all_run_results = []
        previous_average = -math.inf

        while not done:

            params = [(parent_selection_function, None)] * 64

            pool = multiprocessing.Pool()
            run_results = pool.starmap(basic_ea.one_run, params)
            pool.close()

            for r in run_results:
                all_run_results.append(r['final_best_fitness'])
                num_runs += 1
                new_average = statistics.mean(all_run_results)
                if abs(new_average - previous_average) < precision:
                    additions_within_precision += 1
                    if additions_within_precision >= 8:
                        done = True
                        break
                else:
                    additions_within_precision = 0
                previous_average = new_average

        average = statistics.mean(all_run_results)
        print('Optimal runs to achieve {0} precision on {1} selection: {2}'.format(precision, parent_selection_function, num_runs))
        print('Average fitness: {0}'.format(average))



if __name__ == '__main__':
    main()