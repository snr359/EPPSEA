# this script reads the pickled EPPSEA populations in a directory and prints each one, along with its fitness

import sys
import os
import pickle

from eppsea_base import GPTree


if __name__ == '__main__':
    directory = sys.argv[1]
    filenames = list(f for f in os.listdir(directory) if 'gen' in f or f=='final')
    filenames.sort()
    # if final is in the filenames, move it to the end
    if 'final' in filenames:
        filenames.remove('final')
        filenames.append('final')

    # iterate through the populations, unpickling and displaying information about each one
    for filename in filenames:
        print(filename)
        full_filename = '{0}/{1}'.format(directory, filename)
        with open(full_filename, 'rb') as file:
            population = pickle.load(file)
        for p in population:
            p_string = p.get_string()
            p_fitness = p.mo_fitnesses
            print('{0} ||| fitness: {1}'.format(p_string, p_fitness))

