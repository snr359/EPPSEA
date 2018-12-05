# this script finds the lowest dimension that causes cmaes to be unable to solve a coco problem

import configparser
import multiprocessing
import sys

import fitness_functions as ff
import eppsea_cmaes

fitness_function_config = configparser.ConfigParser()
fitness_function_config.read('config/fitness_functions/coco_f1_d5.cfg')

cmaes_config = configparser.ConfigParser()
cmaes_config.read('config/cmaes/config1b.cfg')

runs = 20

def run_basic_cmaes_runner(cmaes_runner):
    result = cmaes_runner.one_run(basic=True)

    return result

def main(start, end):
    for i in range(start,end+1):
        print('------ Testing function {0} -----------'.format(i))
        for d in (2,3,5,10,20):
            print('Testing function {0}, D={1}'.format(i,d))

            fitness_function_config = configparser.ConfigParser()
            fitness_function_config.read('config/fitness_functions/coco_f{0}_d{1}.cfg'.format(i,d))

            cmaes_config = configparser.ConfigParser()
            cmaes_config.read('config/cmaes/config_f{0}_d{1}.cfg'.format(i,d))
            fitness_function_config['fitness function']['genome length'] = str(d)
            fitness_function_config['fitness function']['coco function index'] = str(i)

            cmaes_config['CMAES']['maximum evaluations'] = str(10000*d)
            cmaes_config['CMAES']['population size'] = str(10 * d)

            fitness_functions = ff.generate_coco_functions(fitness_function_config, True)
            cmaess = []
            for fitness_function in fitness_functions:
                cmaes = eppsea_cmaes.CMAES_runner(cmaes_config, fitness_function, None)
                cmaess.append(cmaes)

            params = []
            for cmaes in cmaess:
                params.extend([cmaes]*runs)
            # run all runs
            pool = multiprocessing.Pool(maxtasksperchild=10)
            all_run_results = pool.map(run_basic_cmaes_runner, params)
            pool.close()

            num_solved = len(list(r for r in all_run_results if r.termination_reason == 'target_fitness_hit'))
            percent_solved = round(num_solved / len(all_run_results), 2) * 100

            print('\tFunction {0}, D={1} solved {2}% of problems'.format(i,d,percent_solved))
            if percent_solved < 20:
                print('Stopping function {0} search at {1}'.format(i,d))
                break

if __name__ == '__main__':

    if len(sys.argv) >= 3:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        main(start, end)
    else:
        main(1,24)
