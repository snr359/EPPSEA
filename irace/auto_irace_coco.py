# this script automatically sets up and runs irace for the first 24 coco functions

import configparser
import subprocess
import os
import sys

import tune_basicea_evals

sys.path.insert(0, '../')
import eppsea_basicEA

def main(irace_path):
    # first, set up the fitness functions, using config/basicEA/config5 as the base for the coco functions
    for i in range(1,25):
        new_config = configparser.ConfigParser()
        new_config.read('../config/basicEA/config5.cfg')

        new_config['EA']['fitness function training instances directory'] = '../fitness_functions/coco_f{0}_d10/training'.format(i)
        new_config['EA']['fitness function testing instances directory'] = '../fitness_functions/coco_f{0}_d10/testing'.format(i)
        new_config['fitness function']['coco function index'] = str(i)

        # remove the selection function configs (which will cause a crash anyway, since the relative paths will be different
        new_config.remove_section('basic selection function configs')
        new_config.add_section('basic selection function configs')

        eppsea_basicea_object = eppsea_basicEA.EppseaBasicEA(new_config)
        eppsea_basicea_object.prepare_fitness_functions(new_config)

        # write the config to irace.cfg and tune the eval count for it
        irace_config_file_path = 'irace_config.cfg'
        with open(irace_config_file_path, 'w') as irace_config_file:
            new_config.write(irace_config_file)

        tune_basicea_evals.main(irace_path)

        # edit the path before saving the config file
        new_config['EA']['fitness function training instances directory'] = new_config['EA']['fitness function training instances directory'].replace('../', '')
        new_config['EA']['fitness function testing instances directory'] = new_config['EA']['fitness function testing instances directory'].replace('../', '')

        new_config_file_path = '../config/basicEA/config5_f{0}_d10.cfg'.format(i)
        with open(new_config_file_path, 'w') as new_config_file:
            new_config.write(new_config_file)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide path to irace as argument')
        exit(1)
    irace_path = sys.argv[1]
    main(irace_path)