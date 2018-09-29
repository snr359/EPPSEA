# This script runs the EA in eppsea_basicEA.py for one run, using parameters passed in from the command line
# It prints the final best fitness achieved by the population, and nothing else

import argparse
import configparser
import pickle
import datetime

from eppsea_basicEA import SelectionFunction, FitnessFunction, EA


def get_args():
    # parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config')
    parser.add_argument('--fitness_function_path')

    parser.add_argument('--population_size', type=int)
    parser.add_argument('--offspring_size', type=int)
    parser.add_argument('--mutation_rate', type=float)

    parser.add_argument('--selection_function_config_path', required=False)
    parser.add_argument('--parent_selection', required=False)
    parser.add_argument('--parent_selection_tournament_k', type=int, required=False)
    parser.add_argument('--survival_selection', required=False)
    parser.add_argument('--survival_selection_tournament_k', type=int, required=False)

    parser.add_argument('--generate_configs', action='store_true', required=False)

    args = parser.parse_args()

    if args.selection_function_config_path is None and (args.parent_selection is None or args.survivala_selection is None):
        print('ERROR: parent and survival selection must be defined by either command line argument or config file')
        exit(1)

    if args.parent_selection == 'k_tournament' and args.parent_selection_tournament_k is None:
        print('ERROR: parent_selection is k_tournament but parent_selection_tournament_k is not defined')
        exit(1)
        
    if args.survival_selection == 'k_tournament' and args.survival_selection_tournament_k is None:
        print('ERROR: survival_selection is k_tournament but survival_selection_tournament_k is not defined')
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
    eppsea_ea_config.set('EA', 'population size', str(args.population_size))
    eppsea_ea_config.set('EA', 'offspring size', str(args.offspring_size))
    eppsea_ea_config.set('EA', 'mutation rate', str(args.mutation_rate))

    # create a config object for the parent selection
    selection_config = configparser.ConfigParser()

    # if a selection function configuration file is provided, use it. Otherwise, built one from the command line args
    if args.selection_function_config_path is not None:
        selection_config.read(args.selection_function_config_path)
        selection_function = SelectionFunction()
        selection_function.generate_from_config(selection_config)

    else:
        # set the values of the parent selection config
        selection_config.add_section('selection function')
        selection_config.set('selection function', 'evolved', 'False')
        selection_config.set('selection function', 'file path (for evolved selection)', 'none')
        selection_config.set('selection function', 'display name', '')

        selection_config.set('selection function', 'parent selection type', args.parent_selection)
        if args.parent_selection == 'k_tournament':
            selection_config.set('selection function', 'parent selection tournament k', str(args.parent_selection_tournament_k))
        else:
            selection_config.set('selection function', 'parent selection tournament k', '0')

        selection_config.set('selection function', 'survival selection type', args.survival_selection)
        if args.survival_selection == 'k_tournament':
            selection_config.set('selection function', 'survival selection tournament k', str(args.survival_selection_tournament_k))
        else:
            selection_config.set('selection function', 'survival selection tournament k', '0')

        # create a selection function from the created config
        selection_function = SelectionFunction()
        selection_function.generate_from_config(selection_config)

    # if we are only generating configs, then output the configs and exit
    if args.generate_configs is not None:
        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_eppsea_ea_config_filepath = 'irace_eppsea_basicea_config_{0}.cfg'.format(present_time)
        with open(new_eppsea_ea_config_filepath, 'w') as new_eppsea_ea_config_file:
            eppsea_ea_config.write(new_eppsea_ea_config_file)

        new_selection_config_filepath = 'irace_selection_config_{0}.cfg'.format(present_time)
        with open(new_selection_config_filepath, 'w') as new_selection_config_file:
            selection_config.write(new_selection_config_file)

        exit(0)

    # load the fitness function from the fitness function path
    with open(args.fitness_function_path, 'rb') as fitness_function_file:
        fitness_function = pickle.load(fitness_function_file)

    # create an ea with the fitness function and selection function
    ea = EA(eppsea_ea_config, fitness_function, selection_function)

    # run the ea
    result = ea.one_run()

    # print the mean of the average final best fitnesses, times -1 (so i_race can minimize it)
    print(-1 * result.final_best_fitness)

if __name__ == '__main__':
    main()