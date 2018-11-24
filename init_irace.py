# this script automatically sets up and runs irace

import configparser
import subprocess
import os
import sys
import datetime
import shutil
import argparse
import pickle

from eppsea_basicEA import *

def get_args():
    # parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--irace_path')
    parser.add_argument('-c', '--eppsea_config_path')
    parser.add_argument('-t', '--training_instances_directory')

    parser.add_argument('-e', '--evolved', action='store_true', required=False)
    parser.add_argument('-s', '--evolved_selection_path', required=False)
    parser.add_argument('-C', '--evolved_selection_config_path', required=False)

    parser.add_argument('-T', '--test_against_previous_results', action='store_true', required=False)
    parser.add_argument('-p', '--previous_results_path', required=False)

    args = parser.parse_args()

    if args.evolved and (args.evolved_selection_path is None or args.evolved_selection_config_path is None):
        print('ERROR: to use evolved selection, path to evolved selection and config must be provided')
        exit(1)

    if args.test_against_previous_results and args.previous_results_path is None:
        print('ERROR: to test against previous results, path to previous results must be provided')
        exit(1)

    return args

def main(irace_path, eppsea_config_path, training_instances_directory, using_evolved_selection_function,
         evolved_selection_path, evolved_selection_config_path, test_against_previous_results,
         previous_results_path):
    # first, set up the fitness functions, using config/basicEA/config5 as the base for the coco functions
    eppsea_basicEA_config = configparser.ConfigParser()
    eppsea_basicEA_config.read(eppsea_config_path)

    # write the config to irace.cfg and tune the eval count for it
    irace_config_file_path = 'irace_config.cfg'
    with open(irace_config_file_path, 'w') as irace_config_file:
        eppsea_basicEA_config.write(irace_config_file)

    # set up the irace arguments and call irace
    # get the number of processes
    try:
        num_processes = len(os.sched_getaffinity(0))
    # os.sched_getaffinity may not be available. Fallback to os.cpu_count
    except AttributeError:
        num_processes = os.cpu_count()
    # if os.cpu_count returned none, default to 4
    if num_processes is None:
        num_processes = 4

    # if we are using an evolved selection function, copy it into the base directory
    if using_evolved_selection_function:
        shutil.copy(evolved_selection_path, '_evolved_selection_function')

        config = configparser.ConfigParser()
        config.read(evolved_selection_config_path)
        config['selection function']['file path (for evolved selection)'] = '_evolved_selection_function'
        with open('_evolved_selection_function.cfg', 'w') as config_file:
            config.write(config_file)

        process_args = [irace_path, '--scenario', 'irace_evolved_scenario.txt', '--train-instances-dir', training_instances_directory,
                        '--parallel', str(num_processes)]
    else:
        process_args = [irace_path, '--scenario', 'irace_scenario.txt', '--train-instances-dir', training_instances_directory,
                        '--parallel', str(num_processes)]
    irace_output = subprocess.run(process_args, stdout=subprocess.PIPE, universal_newlines=True).stdout

    # save the irace output
    present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('irace_output_{0}.txt'.format(present_time), 'w') as output_file:
        output_file.write(irace_output)

    # get the final best command line parameters from the irace output
    irace_output_lines = irace_output.split('\n')
    best_configuration_line = irace_output_lines[irace_output_lines.index('# Best configurations as commandlines (first number is the configuration ID; same order as above):') + 1]
    best_configuration_parameters = best_configuration_line.split(' ')[1:]
    while '' in best_configuration_parameters:
        best_configuration_parameters.remove('')

    # output configration files for the new eppsea_basicEA parameters and the selection function
    present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_selection_config_path = 'irace_selection_config_{0}.cfg'.format(present_time)
    new_eppsea_basicea_config_path = 'irace_eppsea_basicea_config_{0}.cfg'.format(present_time)

    process_args = ['python3', 'basicEA_cli.py', '--fitness_function_path', '_', '--base_config', eppsea_config_path,
                    '--selection_config_output_path', new_selection_config_path, '--eppsea_basicea_config_output_path',
                    new_eppsea_basicea_config_path]
    if using_evolved_selection_function:
        process_args.extend(['--selection_function_config_path', '_evolved_selection_function.cfg'])
    process_args.extend(best_configuration_parameters)
    process_args.append('--generate_configs')

    subprocess.run(process_args)

if __name__ == '__main__':
    args = get_args()
    main(args.irace_path, args.eppsea_config_path, args.training_instances_directory, args.evolved,
         args.evolved_selection_path, args.evolved_selection_config_path, args.test_against_previous_results,
         args.previous_results_path)