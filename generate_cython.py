# this module contains code for preparing and compiling python files into Cython files for speedup

import os
import subprocess

def convert_to_pyx(filenames):
    # takes a list of filenames for .py files and converts them to .pyx files. Returns a list of the new pyx files
    filenames_no_extensions = list(f.replace('.py', '') for f in filenames)
    new_filenames = []
    for filename in filenames:
        with open(filename) as file:
            file_text = file.read()
            for other_filename in filenames_no_extensions:
                if 'import {0}'.format(other_filename) in file_text:
                    file_text = file_text.replace('import {0}'.format(other_filename),
                                                  'import {0}_comp as {0}'.format(other_filename))
            new_filename = filename.replace('.py', '_comp.pyx')
            with open(new_filename, 'w') as new_file:
                new_file.write(file_text)
            new_filenames.append(new_filename)
    return new_filenames

def cythonize(filenames, num_cores=None):
    # cythonizes a list of .pyx files, using all available cores
    # if num_cores is none, try to get the number of system cores, or just default to 4
    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except NotImplementedError:
            num_cores = 4

    if num_cores > 1 and num_cores is not None:
        parallel_option = ['parallel={0}'.format(num_cores)]
    else:
        parallel_option = []

    command = ['cythonize', '-3', '--inplace'] + parallel_option + filenames
    subprocess.run(command)
