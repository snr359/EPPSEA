from eppsea_basicEA import SelectionFunction

import math
import random
import statistics
import itertools
import sys
import configparser
import os
import shutil
import multiprocessing
import time
import datetime
import pickle
import json
import uuid

import cocoex

import eppsea_base

from pycma.cma import purecma

class FitnessFunction:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool',
        'mk_landscape': 'bool',
        'coco': 'float'
    }

    def __init__(self):
        self.name = None
        self.genome_type = None
        self.genome_length = None
        self.max_initial_range = None

        self.coco_function_index = None
        self.coco_function_id = None
        self.coco_function = None

        self.started = False

    def __hash__(self):
        return hash(self.id)

    def assign_id(self):
        # assigns a random id to self. Every unique Fitness Function should call this once
        self.id = '{0}_{1}_{2}'.format('FitnessFunction', str(id(self)), str(uuid.uuid4()))

    def generate(self, config):
        # performs all first-time generation for a fitness function. Should be called once per unique fitness function
        self.genome_type = self.genome_types.get(self.name, None)

        self.genome_length = config.getint('fitness function', 'genome length')
        self.max_initial_range = config.getfloat('fitness function', 'max initial range')

        self.coco_function_index = config.getint('fitness function', 'coco function index')

        self.assign_id()

    def start(self):
        # should be called once at the start of each search
        suite = cocoex.Suite('bbob', '', 'dimensions:{0}, function_indices:{1}'.format(self.genome_length, self.coco_function_index))
        self.coco_function = suite.get_problem(self.coco_function_id)
        self.started = True

    def finish(self):
        # should be called once at the end of each EA
        self.coco_function = None
        self.started = False

    def random_genome(self):
        # generates and returns a random genome

        # generate a new genome
        genome = []

        # generate values and append to the genome
        for _ in range(self.genome_length):
            genome.append(random.uniform(-self.max_initial_range, self.max_initial_range))

        # return the new genome
        return genome

    def fitness_target_hit(self):
        return self.coco_function.final_target_hit

    def evaluate(self, genome):
        return self.coco_function(genome)

class ModifiedCMAESRunner(purecma.CMAES):
    def tell(self, arx, fitvals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors"
                a list of candidate solution vectors, presumably from
                calling `ask`. ``arx[k][i]`` is the i-th element of
                solution vector k.
            `fitvals`: `list`
                the corresponding objective function values, to be minimised
        """
        ### bookkeeping and convenience short cuts
        self.counteval += len(fitvals)  # evaluations used within tell
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  # not a copy, xmean is assigned anew later

        ### Sort by fitness
        arx = [arx[k] for k in purecma.argsort(fitvals)]  # sorted arx
        self.fitvals = sorted(fitvals)  # used for termination and display only
        self.best.update(arx[0], self.fitvals[0], self.counteval)

        ### recombination, compute new weighted mean value
        new_arx = random.sample(arx, par.mu)
        self.xmean = purecma.dot(new_arx, par.weights[:par.mu], transpose=True)
        #          = [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                             for i in range(N)]

        ### Cumulation: update evolution paths
        y = purecma.minus(self.xmean, xold)
        z = purecma.dot(self.C.invsqrt, y)  # == C**(-1/2) * (xnew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        for i in range(N):  # update evolution path ps
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = (sum(x**2 for x in self.ps) / N  # ||ps||^2 / N is 1 in expectation
                / (1-(1-par.cs)**(2*self.counteval/par.lam))  # account for initial value of ps
                < 2 + 4./(N+1))  # should be smaller than 2 + ...
        for i in range(N):  # update evolution path pc
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]

        ### Adapt covariance matrix C
        # minor adjustment for the variance loss from hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu * sum(w)
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T, so-called rank-one update
        for k, wk in enumerate(par.weights):  # so-called rank-mu update
            if wk < 0:  # guaranty positive definiteness
                wk *= N * (self.sigma / self.C.mahalanobis_norm(purecma.minus(arx[k], xold)))**2
            self.C.addouter(purecma.minus(arx[k], xold),  # C += wk * cmu * dx * dx^T
                            wk * par.cmu / self.sigma**2)

        ### Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        self.sigma *= purecma.exp(min(1, cn * (sum_square_ps / N - 1) / 2))

class CMAES_runner:
    def __init__(self, config, fitness_function, selection_function):
        self.config = config
        self.fitness_function = fitness_function
        self.selection_function = selection_function

    def one_run(self):
        start = list(random.uniform(-5, 5) for _ in range(10))
        init_sigma = 0.5
        max_evals = 100000

        es = ModifiedCMAESRunner(start, init_sigma, maxfevals=max_evals)

        self.fitness_function.start()

        while not es.stop():
            X = es.ask()  # get a list of sampled candidate solutions
            fit = list(self.fitness_function.evaluate(x) for x in X)
            es.tell(X, fit)  # update distribution parameters


        es.disp(1)
        print('termination by', es.stop())
        print('best f-value =', es.result[1])
        print('solution =', es.result[0])

        if es.best.f < self.fitness_function.evaluate(es.xmean):
            result = [es.best.x, es]
        else:
            result = [es.xmean, es]

        self.fitness_function.finish()

        return result

class EppseaCMAES:
    def __init__(self, config):
        self.config = config

        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_cmaes_" + str(present_time)

        self.results_directory = 'results/eppsea_cmaes/{0}'.format(experiment_name)
        os.makedirs(self.results_directory, exist_ok=True)

        self.log_file_location = '{0}/log.txt'.format(self.results_directory)

        self.using_multiprocessing = config.getboolean('CMAES', 'use multiprocessing')

        self.test_generalization = config.getboolean('CMAES', 'test generalization')
        self.training_runs = config.getint('CMAES', 'training runs')
        self.testing_runs = config.getint('CMAES', 'testing runs')

        self.num_training_fitness_functions = None
        self.num_testing_fitness_functions = None
        self.training_fitness_functions = None
        self.testing_fitness_functions = None

        self.eppsea = None

        self.prepare_fitness_functions(config)

    def prepare_fitness_functions(self, config):
        # generates the fitness functions to be used in the EAs

        # count the number of available function indices
        genome_length = config.getint('fitness function', 'genome length')
        if genome_length not in [2, 3, 5, 10, 20, 40]:
            print('WARNING: genome length {0} may not be supported by coco'.format(genome_length))
        coco_function_index = config.get('fitness function', 'coco function index')
        suite = cocoex.Suite('bbob', '', 'dimensions:{0}, function_indices:{1}'.format(genome_length, coco_function_index))
        coco_ids = list(suite.ids())

        # get the number of training and testing fitness functions to be used
        self.num_training_fitness_functions = config.getint('CMAES', 'num training fitness functions')
        if config.getboolean('CMAES', 'test generalization'):
            self.num_testing_fitness_functions = config.getint('CMAES', 'num testing fitness functions')
            # if we are using coco and testing fitness functions is -1, automatically use remaining instances as test functions
            if self.num_testing_fitness_functions == -1:
                self.num_testing_fitness_functions = len(coco_ids) - self.num_training_fitness_functions
        else:
            self.num_testing_fitness_functions = 0

        self.training_fitness_functions = []
        training_fitness_function_path = config.get('CMAES', 'fitness function training instances directory')
        self.testing_fitness_functions = []
        testing_fitness_function_path = config.get('CMAES', 'fitness function testing instances directory')

        # shuffle coco indeces so there is no bias in assigning training vs testing functions
        if coco_ids is not None:
            random.shuffle(coco_ids)

        if config.getboolean('CMAES', 'generate new fitness functions'):
            for i in range(self.num_training_fitness_functions):
                new_fitness_function = FitnessFunction()
                new_fitness_function.generate(config)
                new_fitness_function.coco_function_id = coco_ids.pop()
                self.training_fitness_functions.append(new_fitness_function)

            for i in range(self.num_testing_fitness_functions):
                new_fitness_function = FitnessFunction()
                new_fitness_function.generate(config)
                new_fitness_function.coco_function_id = coco_ids.pop()
                self.testing_fitness_functions.append(new_fitness_function)

            if config.getboolean('CMAES', 'save generated fitness functions'):
                os.makedirs(training_fitness_function_path, exist_ok=True)
                for i, f in enumerate(self.training_fitness_functions):
                    filepath = '{0}/training{1}'.format(training_fitness_function_path, i)
                    with open(filepath, 'wb') as file:
                        pickle.dump(f, file)

                os.makedirs(testing_fitness_function_path, exist_ok=True)
                for i, f in enumerate(self.testing_fitness_functions):
                    filepath = '{0}/testing{1}'.format(testing_fitness_function_path, i)
                    with open(filepath, 'wb') as file:
                        pickle.dump(f, file)

        else:
            if not os.path.exists(training_fitness_function_path):
                raise Exception('ERROR: Attempting to load fitness functions from non-existent path {0}'.format(
                    training_fitness_function_path))

            training_fitness_function_files = sorted(os.listdir(training_fitness_function_path))
            for filepath in training_fitness_function_files:
                try:
                    full_filepath = '{0}/{1}'.format(training_fitness_function_path, filepath)
                    with open(full_filepath, 'rb') as file:
                        self.training_fitness_functions.append(pickle.load(file))
                except (pickle.PickleError, pickle.PickleError, ImportError, AttributeError):
                    print('Failed to load fitness function at {0}, possibly not a saved fitness function'.format(
                        filepath))
                    pass

                if len(self.training_fitness_functions) == self.num_training_fitness_functions:
                    break

            if config.getboolean('CMAES', 'test generalization'):

                if not os.path.exists(testing_fitness_function_path):
                    raise Exception('ERROR: Attempting to load fitness functions from non-existent path {0}'.format(
                        testing_fitness_function_path))

                testing_fitness_function_files = sorted(os.listdir(testing_fitness_function_path))
                for filepath in testing_fitness_function_files:
                    try:
                        full_filepath = '{0}/{1}'.format(testing_fitness_function_path, filepath)
                        with open(full_filepath, 'rb') as file:
                            self.testing_fitness_functions.append(pickle.load(file))
                    except (pickle.PickleError, pickle.PickleError, ImportError, AttributeError):
                        print('Failed to load fitness function at {0}, possibly not a saved fitness function'.format(
                            filepath))
                        pass

                    if len(self.testing_fitness_functions) == self.num_testing_fitness_functions:
                        break

    def run_eppsea_cmaes(self):
        func = self.testing_fitness_functions[0]
        cmaes = CMAES_runner(None, func, None)
        results = cmaes.one_run()
        print(results)


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = EppseaCMAES(config)
    #shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))
    evaluator.run_eppsea_cmaes()

    # pickle the entire eppsea_basicEA object, and separately the base selection function found and a config file for it, and the final test results
    #evaluator_pickle_path = '{0}/EppseaBasicEA'.format(evaluator.results_directory)
    #with open(evaluator_pickle_path, 'wb') as pickle_file:
    #    pickle.dump(evaluator, pickle_file)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    main(config_path)