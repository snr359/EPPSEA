import configparser

import eppsea_cmaes

for i in range(1, 25):
    for config_path in ('config/cmaes/config{0}.cfg', 'config/cmaes/config{0}b.cfg'):
        print('Running with config at {0}'.format(config_path))
        config = configparser.ConfigParser()
        config.read(config_path)

        evaluator = eppsea_cmaes.EppseaCMAES(config)
        fitness_functions = evaluator.training_fitness_functions + evaluator.testing_fitness_functions
        basic_cmaess = evaluator.get_basic_cmaes_runners(fitness_functions)
        basic_cmaess_results = evaluator.run_basic_cmaes_runners(basic_cmaess, True)
        for f in fitness_functions:
            f_results = list(r for r in basic_cmaess_results if r.fitness_function_id == f.id)
            num_solved = len(list(r for r in f_results if r.termination_reason == 'target_fitness_hit'))
            percentage_solved = round(num_solved / len(f_results), 2)
            print('Basic CMAES solved fitness function {0} {1}% of the time'.format(f.display_name, percentage_solved))