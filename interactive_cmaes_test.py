import sys
import configparser
import statistics

import eppsea_base
import eppsea_cmaes

from eppsea_basicEA import SelectionFunction

def log(s):
    print(s)
    with open('output.txt', 'a') as file:
        file.write('{0}\n'.format(s))
    
def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    cmaes = eppsea_cmaes.EppseaCMAES(config)
    eppsea_config = config.get('CMAES', 'base eppsea config path')
    eppsea = eppsea_base.Eppsea(eppsea_config)
    
    eppsea.start_evolution()

    eppsea_selection_functions = []
    
    while True:
        print('EPPSEA functions:')
        for i,f in enumerate(eppsea.population):
            print('{0} --- {1}'.format(i, f.get_string()))
        user_input = input('select a function to add, "u" to undo last selection, "s" to start testing now\n')
        try:
            if 0 <= int(user_input) < len(eppsea.population):
                eppsea_selection_functions.append(eppsea.population[int(user_input)])
        except ValueError:
            pass
        if user_input == 'u':
            eppsea_selection_functions.pop()
        if user_input == 's':
            break

    if len(eppsea_selection_functions) > 0:
        new_selection_function = eppsea_base.EppseaSelectionFunction(eppsea_selection_functions[0])
        new_selection_function.gp_trees[0].root.operation = 'fitness'
        new_selection_function.gp_trees[0].root.children = []
        new_selection_function.gp_trees[0].selection_type = 'truncation'

        eppsea_selection_functions.append(new_selection_function)

    selection_functions = []
    for e in eppsea_selection_functions:
        selection_function = SelectionFunction()
        selection_function.generate_from_eppsea_individual(e)
        selection_functions.append(selection_function)

    fitness_functions = cmaes.testing_fitness_functions + cmaes.training_fitness_functions

    eas = cmaes.get_cmaes_runners(fitness_functions, selection_functions)
    ea_results = cmaes.run_cmaes_runners(eas, True)
    
    basic_cmaess = cmaes.get_basic_cmaes_runners(fitness_functions)
    basic_cmaess_results = cmaes.run_basic_cmaes_runners(basic_cmaess, True)

    all_results = ea_results + basic_cmaess_results

    basic_average_best_fitnesses = dict()
    for f in fitness_functions:
        f_results = list(r for r in basic_cmaess_results if r.fitness_function_id == f.id)
        basic_average_best_fitnesses[f.id] = statistics.mean(r.final_best_fitness for r in f_results)

    for f in fitness_functions:
        log('Results on fitness function {0} with id {1}'.format(f.display_name, f.id))
        f_results = list(r for r in all_results if r.fitness_function_id == f.id)
        for s in selection_functions:
            log('\tResults for EPPSEA member: {0}'.format(s.eppsea_selection_function.get_string()))
            s_results = list(r for r in f_results if r.selection_function_id == s.id)
            percentage_better = len(list(r for r in s_results if r.final_best_fitness <= basic_average_best_fitnesses[f.id])) / len(s_results)
            percentage_better = round(100 * percentage_better, 2)
            log('\t\tThis selection function beat basic cmaes {0}% of the time'.format(percentage_better))

    for i in (3,5,10,20,50,100, 200, 500, 1000, 2000):
        if i < cmaes.testing_runs:
            log('\n----------------------------------------------------------------------------\n')
            log('Same reports if only the first {0} results are taken:'.format(i))
            basic_average_best_fitnesses = dict()
            for f in fitness_functions:
                f_results = list(r for r in basic_cmaess_results if r.fitness_function_id == f.id)[:i]
                basic_average_best_fitnesses[f.id] = statistics.mean(r.final_best_fitness for r in f_results)

            for f in fitness_functions:
                log('Results on fitness function {0} with id {1}'.format(f.display_name, f.id))
                f_results = list(r for r in all_results if r.fitness_function_id == f.id)
                for s in selection_functions:
                    log('\tResults for EPPSEA member: {0}'.format(s.eppsea_selection_function.get_string()))
                    s_results = list(r for r in f_results if r.selection_function_id == s.id)[:i]
                    percentage_better = len(list(r for r in s_results if r.final_best_fitness <= basic_average_best_fitnesses[f.id])) / len(s_results)
                    percentage_better = round(100 * percentage_better, 2)
                    log('\t\tThis selection function beat basic cmaes {0}% of the time'.format(percentage_better))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_paths = sys.argv[1:]

    for config_path in config_paths:
        main(config_path)