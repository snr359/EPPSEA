# this script displays which results directories correspond to which coco bbob functions for eppsea_cmaes

import configparser
import os
import shutil

def main():
    # first, try to get rid of directories that do not have valid results
    base_result_directory = 'results/eppsea_basicEA'
    results_directories = list('{0}/{1}'.format(base_result_directory, r) for r in os.listdir(base_result_directory))
    empty_results_directories = []
    for r in results_directories:
        if not any('.png' in filename for filename in os.listdir(r)) and not os.path.exists('{0}/final_results'.format(r)):
            empty_results_directories.append(r)
    if empty_results_directories:
        print('The following directories do not seem to have final results files:')
        for r in sorted(empty_results_directories):
            print(r)
        print('Delete them? (y/n)')
        while True:
            choice = input()
            if choice == 'y':
                for r in empty_results_directories:
                    shutil.rmtree(r)
                break
            elif choice == 'n':
                break
    results_directories = list('{0}/{1}'.format(base_result_directory, r) for r in os.listdir(base_result_directory))

    function_results = dict()
    function_paths = []
    for i in range(1,25):
            function_path = 'config/fitness_functions/coco_f{0}_d10.cfg'.format(i)
            function_results[function_path] = []
            function_paths.append(function_path)

    for r in results_directories:
        config_path = '{0}/config.cfg'.format(r)
        final_results_path = '{0}/final_results'.format(r)
        if os.path.exists(config_path) and os.path.exists(final_results_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            function_path = config.get('EA', 'fitness function config path')
            if function_path in function_results:
                function_results[function_path].append(r)

    for f in function_paths:
        print('Results directories for fitness function at {0}'.format(f))
        for rr in sorted(function_results[f]):
            print('\t' + rr)
        print('-----------------------------------------------------------------------')

if __name__ == '__main__':
    main()