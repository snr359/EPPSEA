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

def run_cmaes_runner(cmaes_runner):
    final_best, final_cmaes, term_conditions = cmaes_runner.one_run()
    result = CMAES_Result()
    result.final_best_fitness = final_cmaes.best.f
    result.eval_counts = final_cmaes.counteval

    result.fitness_function = cmaes_runner.fitness_function
    result.fitness_function_id = cmaes_runner.fitness_function.id

    result.selection_function = cmaes_runner.selection_function
    result.selection_function_id = cmaes_runner.selection_function.id

    if 'ftarget' in term_conditions:
        result.termination_reason = 'target_fitness_hit'
    elif 'maxfevals' in term_conditions:
        result.termination_reason = 'maximum_evaluations_reached'
    elif 'tolfun' in term_conditions or 'tolx' in term_conditions:
        result.termination_reason = 'fitness_convergence'

    return result


class CMAES_Result:
    # a class for holding the results of a single run of a cmaes
    def __init__(self):
        self.eval_counts = []
        self.final_best_fitness = None
        self.termination_reason = None

        self.selection_function_name = None
        self.selection_function_id = None

        self.fitness_function_name = None
        self.fitness_function_id = None


class CMAES_ResultCollection:
    # a class for holding the results of several cmaes runs
    def __init__(self, results=None):
        self.results = []
        self.fitness_functions = list()
        self.selection_functions = list()

        if results is not None:
            self.add(results)

    def add(self, new_results):
        # adds a list of new results to the result collection
        self.results.extend(new_results)
        for r in new_results:
            if not any(r.selection_function.id == s.id for s in self.selection_functions):
                self.selection_functions.append(r.selection_function)
            if not any(r.fitness_function.id == f.id for f in self.fitness_functions):
                self.fitness_functions.append(r.fitness_function)

    def get_eval_counts(self):
        all_counts = set()
        for r in self.results:
            counts = r.eval_counts
            all_counts.update(counts)
        return sorted(list(all_counts))

    def filter(self, selection_function=None, fitness_function=None):
        # this returns a new instance of EAResultCollection, with a subset of the EAResults in results
        # if selection_function is provided, it keeps only the results with a matching selection_function
        # if fitness_function is provided, it keeps only the results with a matching selection_function

        # filter by selection function id, fitness function id, or both
        if selection_function is not None and fitness_function is not None:
            filtered_results = list(r for r in self.results if r.selection_function.id == selection_function.id
                                    and r.fitness_function.id == fitness_function.id)
        elif selection_function is not None:
            filtered_results = list(r for r in self.results if r.selection_function.id == selection_function.id)
        elif fitness_function is not None:
            filtered_results = list(r for r in self.results if r.fitness_function.id == fitness_function.id)
        else:
            filtered_results = list(self.results)

        # create a new result collection with the filtered results and return it
        new_collection = CMAES_ResultCollection(filtered_results)
        return new_collection


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

class ModifiedCMAES(purecma.CMAES):
    def tell_pop(self, arx, fitvals, population, selection_function):
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
        self.counteval += len(population)  # evaluations used within tell
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  # not a copy, xmean is assigned anew later

        ### Sort by fitness
        arx = [arx[k] for k in purecma.argsort(fitvals)]  # sorted arx
        self.fitvals = sorted(fitvals)  # used for termination and display only
        self.best.update(arx[0], self.fitvals[0], self.counteval)

        ### recombination, compute new weighted mean value
        # new_arx = random.sample(arx, par.mu)
        selected_members = selection_function.eppsea_selection_function.select(population, par.mu)
        selected_arx = list(p.genome for p in selected_members)
        self.xmean = purecma.dot(selected_arx, par.weights[:par.mu], transpose=True)
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
            self.C.addouter(purecma.minus(arx[k], xold), wk * par.cmu / self.sigma**2)  # C += wk * cmu * dx * dx^T


        ### Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        self.sigma *= purecma.exp(min(1, cn * (sum_square_ps / N - 1) / 2))

    def stop(self):
        res = {}
        if self.counteval <= 0:
            return res
        if self.counteval >= self.maxfevals:
            res['maxfevals'] = self.maxfevals
        if self.fitness_function.coco_function.final_target_hit:
            res['ftarget'] = self.fitvals[0]
        if self.C.condition_number > 1e14:
            res['condition'] = self.C.condition_number
        if self.sigma > 1e140:
            res['sigma'] = self.sigma
        if len(self.fitvals) > 1 and (self.fitvals[0] == float('inf') or self.fitvals[-1] - self.fitvals[0] < 1e-12):
            res['tolfun'] = 1e-12
        if self.sigma * max(self.C.eigenvalues)**0.5 < 1e-11:
            # remark: max(D) >= max(diag(C))**0.5
            res['tolx'] = 1e-11
        return res

class CMAES_runner:
    def __init__(self, config, fitness_function, selection_function):
        self.config = config
        self.fitness_function = fitness_function
        self.selection_function = selection_function

    class Popi:
        def __init__(self):
            self.genome = None
            self.fitness = None
            self.birth_generation = None

    def one_run(self, basic=False):
        start = list(random.uniform(-5, 5) for _ in range(self.fitness_function.genome_length))
        init_sigma = 0.5
        max_evals = self.config.getint('CMAES', 'maximum evaluations')

        es = ModifiedCMAES(start, init_sigma, maxfevals=max_evals, popsize=100)
        es.fitness_function = self.fitness_function

        self.fitness_function.start()

        generation = 0

        while not es.stop():
            X = es.ask()  # get a list of sampled candidate solutions
            fitness_values = list(self.fitness_function.evaluate(x) for x in X)
            population = []
            for x, fit in zip(X, fitness_values):
                new_popi = self.Popi()
                new_popi.genome = x
                new_popi.fitness = -1 * fit #eppsea assumes fitness maximization
                new_popi.birth_generation = generation
                population.append(new_popi)
            if basic:
                es.tell(X, fitness_values)
            else:
                es.tell_pop(X, fitness_values, population, self.selection_function)  # update distribution parameters
            generation += 1

        es.disp(1)
        term_conditions = es.stop()
        print('termination by', term_conditions)
        print('best f-value =', es.result[1])
        print('solution =', es.result[0])

        if es.best.f < self.fitness_function.evaluate(es.xmean):
            result = [es.best.x, es, term_conditions]
        else:
            result = [es.xmean, es, term_conditions]

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

        if config.get('CMAES', 'eppsea fitness assignment method') == 'best fitness reached' or config.get('CMAES', 'eppsea fitness assignment method') == 'adaptive':
            self.eppsea_fitness_assignment_method = 'best_fitness_reached'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'proportion hitting target fitness':
            self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'evals to target fitness':
            self.eppsea_fitness_assignment_method = 'evals_to_target_fitness'
        else:
            raise Exception('ERROR: eppsea fitness assignment method {0} not recognized!'.format(config.get('EA', 'eppsea fitness assignment method')))

        self.eppsea = None

        self.prepare_fitness_functions(config)

    def log(self, message):
        print(message)
        with open(self.log_file_location, 'a') as log_file:
            log_file.write(message + '\n')

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

    def get_cmaes_runners(self, fitness_functions, selection_functions):
        # this prepares and returns a list of cmaes_runners for the provided selection functions
        # selection_functions is a list of tuples of the form (selection_function_name, selection_function),
        # where, if the selection_function_name is 'eppsea_selection_function', then selection_function is
        # expected to be an eppsea selection function

        result = list()

        for selection_function in selection_functions:
            for fitness_function in fitness_functions:
                result.append(CMAES_runner(self.config, fitness_function, selection_function))

        return result

    def run_cmaes_runners(self, eas, is_testing):
        # runs each of the eas for the configured number of runs, returning one result_holder for each ea
        all_run_results = []

        if is_testing:
            runs = self.testing_runs
        else:
            runs = self.training_runs

        if self.using_multiprocessing:
            # setup parameters for multiprocessing
            params = []
            for ea in eas:
                params.extend([ea]*runs)
            # run all runs
            pool = multiprocessing.Pool()
            results = pool.map(run_cmaes_runner, params)
            pool.close()

            # split up results by ea
            for i in range(len(eas)):
                start = i * runs
                stop = (i + 1) * runs

                run_results = results[start:stop]

                all_run_results.extend(run_results)

        else:
            for ea in eas:
                for r in range(runs):
                    all_run_results.append(run_cmaes_runner(ea))

        return CMAES_ResultCollection(all_run_results)

    def evaluate_eppsea_population(self, eppsea_population, is_testing):
        # evaluates a population of eppsea individuals and assigns fitness values to them
        if is_testing:
            fitness_functions = self.testing_fitness_functions
        else:
            fitness_functions = self.training_fitness_functions

        selection_functions = []
        for e in eppsea_population:
            selection_function = SelectionFunction()
            selection_function.generate_from_eppsea_individual(e)
            selection_functions.append(selection_function)

        eas = self.get_cmaes_runners(fitness_functions, selection_functions)
        ea_results = self.run_cmaes_runners(eas, False)
        self.assign_eppsea_fitness(selection_functions, ea_results)

    def assign_eppsea_fitness(self, selection_functions, ea_results):
        for s in selection_functions:
            s_results = ea_results.filter(selection_function=s)
            if self.config.get('CMAES', 'eppsea fitness assignment method') == 'adaptive':
                if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                    if len(list(r for r in s_results.results if r.termination_reason == 'target_fitness_hit')) / len (s_results.results) >= 0.1:
                        self.log('At eval count {0}, eppsea fitness assignment changed to proportion_hitting_target_fitness'.format(self.eppsea.gp_evals))
                        self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
                        for p in self.eppsea.population:
                            p.fitness = -math.inf
                        self.eppsea.gens_since_avg_fitness_improvement = 0
                        self.eppsea.gens_since_best_fitness_improvement = 0
                        self.eppsea.highest_average_fitness = -math.inf
                        self.eppsea.highest_best_fitness = -math.inf
                if self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                    if len(list(r for r in s_results.results if r.termination_reason == 'target_fitness_hit')) / len(s_results.results) >= .95:
                        self.log('At eval count {0}, eppsea fitness assignment changed to evals_to_target_fitness'.format(self.eppsea.gp_evals))
                        self.eppsea_fitness_assignment_method = 'evals_to_target_fitness'
                        for p in self.eppsea.population:
                            p.fitness = -math.inf
                        self.eppsea.gens_since_avg_fitness_improvement = 0
                        self.eppsea.gens_since_best_fitness_improvement = 0
                        self.eppsea.highest_average_fitness = -math.inf
                        self.eppsea.highest_best_fitness = -math.inf

        # loop through the selection functions containing the eppsea individuals
        for s in selection_functions:
            # filter out the ea runs associated with this individual
            s_results = ea_results.filter(selection_function=s)

            if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                # loop through all fitness functions to get average final best fitnesses
                average_final_best_fitnesses = []
                for fitness_function in s_results.fitness_functions:
                    fitness_function_results = s_results.filter(fitness_function=fitness_function)
                    final_best_fitnesses = (r.final_best_fitness for r in fitness_function_results.results)
                    average_final_best_fitnesses.append(statistics.mean(final_best_fitnesses))
                # assign fitness as -1 times the average of the average final best fitnesses or, if multiobjective ea is on, the list of average final best fitnesses
                s.eppsea_selection_function.fitness = -1 * statistics.mean(average_final_best_fitnesses)

            elif self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                # assign the fitness as the proportion of runs that hit the target fitness
                s.eppsea_selection_function.fitness = len(list(r for r in s_results.results if r.termination_reason == 'target_fitness_hit')) / len (s_results.results)

            elif self.eppsea_fitness_assignment_method == 'evals_to_target_fitness':
                # loop through all fitness functions to get average evals to target fitness
                all_final_evals = []
                for fitness_function in s_results.fitness_functions:
                    fitness_function_results = s_results.filter(fitness_function=fitness_function)
                    final_evals = list(max(r.evals) for r in fitness_function_results if r.termination_reason == 'target_fitness_hit')
                    # for the runs where the target fitness was not hit, use an eval count equal to twice the maximum count
                    for r in fitness_function_results:
                        if r.termination_reason != 'target_fitness_hit':
                            final_evals.append(2 * max(r.evals) for r in fitness_function_results)
                    all_final_evals.append(statistics.mean(final_evals))
                # assign fitness as -1 * the average of final eval counts
                s.eppsea_selection_function.fitness = -1 * statistics.mean(all_final_evals)
            else:
                raise Exception('ERROR: fitness assignment method {0} not recognized by eppsea_basicEA'.format(self.eppsea_fitness_assignment_method))

    def run_eppsea_cmaes(self):
        print('Now starting EPPSEA')
        start_time = time.time()

        eppsea_config = self.config.get('CMAES', 'base eppsea config path')
        eppsea = eppsea_base.Eppsea(eppsea_config)
        self.eppsea = eppsea

        eppsea.start_evolution()

        while not eppsea.evolution_finished:
            self.evaluate_eppsea_population(eppsea.new_population, False)
            eppsea.next_generation()

        best_selection_function = eppsea.final_best_member

        #print('Running final tests')
        #self.final_test_results = self.test_against_basic_cmaes(best_selection_function)

        #self.log('Running Postprocessing')
        #postprocess_results = self.postprocess(self.final_test_results)
        #self.log('Postprocess results:')
        #self.log(postprocess_results)

        eppsea_base_results_path = eppsea.results_directory
        shutil.copytree(eppsea_base_results_path, self.results_directory + '/base')
        end_time = time.time() - start_time
        self.log('Total time elapsed: {0}'.format(end_time))


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