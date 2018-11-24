# this script displays which results directories correspond to which coco bbob functions for eppsea_cmaes

import configparser
import os
import shutil

def main():
    # first, try to get rid of directories that do not have valid results
    base_result_directory = 'results/eppsea_cmaes'
    results_directories = list('{0}/{1}'.format(base_result_directory, r) for r in os.listdir(base_result_directory))
    empty_results_directories = []
    for r in results_directories:
        if not os.path.exists('{0}/final_results'.format(r)):
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
    for i in range(1,25):
        for n in (5,10):
            function_path = 'config/fitness_functions/coco_f{0}_d{1}.cfg'.format(i,n)
            function_results[function_path] = []

    for r in results_directories:
        config_path = '{0}/config.cfg'.format(r)
        config = configparser.ConfigParser()
        config.read(config_path)
        function_path = config.get('CMAES', 'fitness function config path')
        if function_path in function_results:
            function_results[function_path].append(r)

    for f, r in function_results.items():
        print('Results directories for fitness function at {0}'.format(f))
        for rr in sorted(r):
            print('\t' + rr)
        print('-----------------------------------------------------------------------')

if __name__ == '__main__':
    main()