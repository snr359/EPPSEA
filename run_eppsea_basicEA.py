# this script runs eppesa_basicEA for each config file passed into it
# it will attempt to cythonize the code before running

import traceback
import sys
import os
import datetime

import generate_cython

def main(config_paths):
    # try to get the compiled (cythonized) code first
    try:
        import eppsea_basicEA_comp as eppsea_basicEA
        print('Imported compiled eppsea code')
    except ImportError:
        print('Compiled eppsea code not found. Attempting to compile')
        try:
            generate_cython.convert_to_pyx(['eppsea_basicEA.py', 'eppsea_base.py'])
            generate_cython.cythonize(['*.pyx'])
            import eppsea_basicEA_comp as eppsea_basicEA
            print('Compiled and imported eppsea code')
        except:
            print('Failed to compile and import. Falling back to pure python')
            import eppsea_basicEA

    # run the code for each configuration file
    for config_path in config_paths:
        if not os.path.isfile(config_path):
            print('No config file found at {0}'.format(config_path))
            continue
        try:
            print('Running eppsea_basicEA with config file {0}'.format(config_path))
            eppsea_basicEA.main(config_path)
        except :
            print('eppsea_basicEA failed. Printing stacktrace into error.txt ')
            with open('error.txt', 'a') as error_file:
                present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                error_file.write('======== STACK TRACE {0} =============\n'.format(present_time))
                stack_trace = traceback.format_exc()
                error_file.write(stack_trace)

if __name__ == '__main__':
    config_paths = sys.argv[1:]
    if len(config_paths) == 0:
        print('Provide list of config paths as arguments')
    else:
        main(config_paths)