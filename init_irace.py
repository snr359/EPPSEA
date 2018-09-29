# this script automatically sets up and runs irace

import configparser
import subprocess
import os
import sys
import datetime

def main(irace_path, base_config_path, training_instances_directory, using_evolved_selection_function):
    # first, set up the fitness functions, using config/basicEA/config5 as the base for the coco functions
    config = configparser.ConfigParser()
    config.read(base_config_path)

    # write the config to irace.cfg and tune the eval count for it
    irace_config_file_path = 'irace_config.cfg'
    with open(irace_config_file_path, 'w') as irace_config_file:
        config.write(irace_config_file)

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

    process_args = [irace_path, '--scenario', 'irace_scenario.txt', '--train-instances-dir', training_instances_directory,
                    '--parallel', str(num_processes)]
    irace_output = subprocess.run(process_args, stdout=subprocess.PIPE, universal_newlines=True).stdout

    # save the irace output
    present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('irace_output_{0}.txt'.format(present_time), 'w') as output_file:
        output_file.write(irace_output)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide the following arguments:')
        print('Path to irace executable')
        print('Path to base eppsea_basicEA configuration file')
        print('Path to irace training instances')
        print('"evolved" if using eppsea evolved selection function')
        exit(1)
    irace_path = sys.argv[1]
    base_config_path = sys.argv[2]
    training_instances_path = sys.argv[3]
    if len(sys.argv) >= 5 and sys.argv[4] == 'evolved':
        using_evolved_selection_function = True
    else:
        using_evolved_selection_function = False
    main(irace_path, base_config_path, training_instances_path, using_evolved_selection_function)