# This script runs irace (http://iridia.ulb.ac.be/irace/) to find the optimal parameters for the EA used in
# the eppsea_basicEA script. It takes as a paramater an eppsea_basicEA configuration, from which it determines
# some EA settings, and uses irace to find values for other settings.

import sys
sys.path.insert(0, '../')

import argparse
import configparser
import pickle

from eppsea_basicEA import SelectionFunction, FitnessFunction, EA


def get_args():
    # parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config')
    parser.add_argument('--fitness_function_path')
    parser.add_argument('--population_size', type=int)
    parser.add_argument('--offspring_size', type=int)
    parser.add_argument('--mutation_rate', type=float)
    parser.add_argument('--survival_selection')
    parser.add_argument('--parent_selection')
    parser.add_argument('--parent_selection_tournament_k', type=int, required=False)

    args = parser.parse_args()

    if args.survival_selection not in ('random', 'elitist_random', 'truncation'):
        print('ERROR: survival_selection {0} not recognized'.format(args.survival_selection))
        exit(1)

    if args.parent_selection == 'k_tournament' and args.parent_selection_tournament_k is None:
        print('ERROR: parent_selection isk_tournament but parent_selection_tournament_k is not defined')
        exit(1)

    return args

def main():
    # get the command-line arguments
    args = get_args()

    # create a config object for the EA
    eppsea_ea_config = configparser.ConfigParser()

    # read the base config
    eppsea_ea_config.read(args.base_config)

    # change the values of the config according to the parameters
    eppsea_ea_config.set('EA','population size', str(args.population_size))
    eppsea_ea_config.set('EA', 'offspring size', str(args.offspring_size))
    eppsea_ea_config.set('EA', 'mutation rate', str(args.mutation_rate))
    eppsea_ea_config.set('EA', 'survival selection', args.survival_selection)

    # create a config object for the parent selection
    parent_selection_config = configparser.ConfigParser()

    # set the values of the parent selection config
    parent_selection_config.add_section('selection function')
    parent_selection_config.set('selection function', 'type', args.parent_selection)
    parent_selection_config.set('selection function', 'name', args.parent_selection)
    if args.parent_selection == 'k_tournament':
        parent_selection_config.set('selection function', 'tournament k', str(args.parent_selection_tournament_k))
    else:
        parent_selection_config.set('selection function', 'tournament k', '0')

    # create a selection function from the created config
    selection_function = SelectionFunction()
    selection_function.generate_from_config(parent_selection_config)

    # load the fitness function from the fitness function path
    with open(args.fitness_function_path, 'rb') as fitness_function_file:
        fitness_function = pickle.load(fitness_function_file)

    # create an ea with the fitness function and selection function
    ea = EA(eppsea_ea_config, fitness_function, selection_function)

    # run the ea
    result = ea.one_run()

    # print the mean of the average final best fitnesses, times -1 (so i_race can minimize it)
    print(-1 * result['final_best_fitness'])

if __name__ == '__main__':
    main()