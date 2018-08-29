# this function finds a good upper count for evals needed before basicea fails to increase in performance.
# it uses irace to do so

import configparser
import subprocess
import os
import sys

import scipy.stats

def get_irace_result_commands(irace_output):
    # parses the output of irace to get the best resulting command-line arguments for irace_basicEA.py
    # split the output into lines
    lines = irace_output.split('\n')
    # find the line before the best result
    i = 0
    while '# Best configurations as commandlines (first number is the configuration ID; same order as above):' not in lines[i]:
        i += 1
    # the best line is the immediate next line
    best_line = lines[i+1]
    best_line = best_line.strip()

    # strip out the configuration id
    first_space_index = best_line.index(' ')
    best_line = best_line[first_space_index+1:]
    best_line = best_line.strip()

    return best_line

def run_irace(irace_path, eval_count, training_instances_directory):
    # edits the irace_config.cfg to use the specified eval count and runs irace with the new eval count
    # open and edit the config file
    config = configparser.ConfigParser()
    config.read('irace_config.cfg')

    config['EA']['maximum evaluations'] = str(eval_count)

    with open('irace_config.cfg', 'w') as file:
        config.write(file)

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

    process_args = [irace_path, '--train-instances-dir', training_instances_directory, '--parallel', str(num_processes)]
    irace_output = subprocess.run(process_args, stdout=subprocess.PIPE, universal_newlines=True).stdout

    # return irace output
    return irace_output

def test_parameters(command_line_params, num_runs, training_instance_directory):
    # runs the basicEA with the given command line parameters for the specified number of runs
    # returns one list of results for each training instance, contained in a dict
    # set up the process arguments
    base_process_args = ['python3', 'irace_basicEA.py', '--base_config', 'irace_config.cfg']
    base_process_args += command_line_params.split(' ')

    results = dict()

    # loop through all training instances
    for training_instance in os.listdir(training_instance_directory):
        results[training_instance] = []
        process_args = base_process_args[:] + ['--fitness_function_path', training_instance_directory + '/' + training_instance]

        # run the basicEA
        for _ in range(num_runs):
            result = float(subprocess.run(process_args, stdout=subprocess.PIPE, universal_newlines=True).stdout)
            results[training_instance].append(result)

    return results

def significant_difference(sample1, sample2):
    # returns true if a t-test between list_a and list_b indicates they are significantly different, with p<.01
    _, p = scipy.stats.ttest_rel(sample1, sample2)
    return p < 0.01

def main(irace_path):
    # does an exponential search to find the maximum number of evals for basicEA, beyond which no benefit is found
    print('Starting exponential search for maximum basicEA evals needed')
    start_evals = 200
    config = configparser.ConfigParser()
    config.read('irace_config.cfg')
    training_instances_directory = config['EA']['fitness function training instances directory']

    # first, find a ceiling and floor for a binary search, starting with 1000
    print('Running irace with {0} evals'.format(start_evals))
    irace_output = run_irace(irace_path, start_evals, training_instances_directory)
    best_command_line_args = get_irace_result_commands(irace_output)

    evals_results = dict()
    evals_irace_commands = dict()
    print('Testing basicEA with {0} evals'.format(start_evals))
    evals_results[start_evals] = test_parameters(best_command_line_args, 30, training_instances_directory)
    evals_irace_commands[start_evals] = best_command_line_args

    ceiling_found = False
    previous_evals = start_evals

    print('Beginning search for ceiling')
    while not ceiling_found:
        new_evals = previous_evals * 2
        print('Running irace with {0} evals'.format(new_evals))
        irace_output = run_irace(irace_path, new_evals, training_instances_directory)
        best_command_line_args = get_irace_result_commands(irace_output)
        print('Testing basicEA with {0} evals'.format(new_evals))
        evals_results[new_evals] = test_parameters(best_command_line_args, 30, training_instances_directory)
        evals_irace_commands[new_evals] = best_command_line_args

        # check if the results for any of the training instances are significantly different
        any_different = False
        for training_instance in os.listdir(training_instances_directory):
            previous_results = evals_results[previous_evals][training_instance]
            new_results = evals_results[new_evals][training_instance]
            if significant_difference(previous_results, new_results):
                any_different = True
                break

        # if none of the results are significantly different, we have found a cieling
        if any_different:
            print('Results for at least one training instance are significantly different')
            previous_evals = new_evals
        else:
            print('No results for any training instance are significantly different. Ceiling found at {0} evals'.format(new_evals))
            ceiling_found = True

    # now do a binary search to find the resulting eval count that produces no different (within a window of 100 evals)
    floor = previous_evals
    ceiling = new_evals

    while ceiling - floor > 100:
        new_evals = int((ceiling + floor) / 2)
        print('Running irace with {0} evals'.format(new_evals))
        irace_output = run_irace(irace_path, new_evals, training_instances_directory)
        best_command_line_args = get_irace_result_commands(irace_output)
        print('Testing basicEA with {0} evals'.format(new_evals))
        evals_results[new_evals] = test_parameters(best_command_line_args, 30, training_instances_directory)
        evals_irace_commands[new_evals] = best_command_line_args

        # check if the results for any of the training instances are significantly different
        any_different = False
        for training_instance in os.listdir(training_instances_directory):
            ceil_results = evals_results[ceiling][training_instance]
            new_results = evals_results[new_evals][training_instance]
            if significant_difference(ceil_results, new_results):
                any_different = True
                break

        # if none of the results are significantly different, lower the ceiling. Otherwise, raise the floor
        if any_different:
            print('Results for at least one training instance are significantly different. Raising floor...')
            floor = new_evals
        else:
            print('No results for any training instance are significantly different. Lowering ceiling...')
            ceiling = new_evals

    # print the final ceiling as the eval count
    print('Final eval count found is {0} evals'.format(ceiling))
    print('Final command line args for final ceiling:')
    print(evals_irace_commands[ceiling])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide path to irace as argument')
        exit(1)
    irace_path = sys.argv[1]
    main(irace_path)