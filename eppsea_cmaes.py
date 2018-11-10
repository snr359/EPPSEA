from eppsea_basicEA import SelectionFunction

import math
import statistics
import sys
import configparser
import os
import shutil
import multiprocessing
import time
import datetime
import pickle
import subprocess

import eppsea_base
import fitness_functions as ff

from pycma.cma import purecma

def run_cmaes_runner(cmaes_runner):
    result = cmaes_runner.one_run()

    return result

def run_basic_cmaes_runner(cmaes_runner):
    result = cmaes_runner.one_run(basic=True)

    return result

def export_results(results, file_path):
    # exports a list of CMAES_Result objects to a file_path for postprocessing
    run_dicts = list(r.export() for r in results)
    with open(file_path, 'wb') as file:
        pickle.dump(run_dicts, file)

class CMAES_Result:
    # a class for holding the results of a single run of a cmaes
    def __init__(self):
        self.eval_counts = None
        self.fitnesses = None
        self.average_fitnesses = None
        self.best_fitnesses = None
        self.final_best_fitness = None
        self.termination_reason = None

        self.selection_function_display_name = None
        self.selection_function_id = None
        self.selection_function_was_evolved = None
        self.selection_function_eppsea_string = None

        self.fitness_function_display_name = None
        self.fitness_function_id = None

    def export(self):
        info_dict = dict()

        info_dict['eval_counts'] = self.eval_counts
        info_dict['fitnesses'] = self.fitnesses
        info_dict['average_fitnesses'] = self.average_fitnesses
        info_dict['best_fitnesses'] = self.best_fitnesses
        info_dict['final_best_fitness'] = self.final_best_fitness
        info_dict['termination_reason'] = self.termination_reason

        info_dict['selection_function_display_name'] = self.selection_function_display_name
        info_dict['selection_function_id'] = self.selection_function_id
        info_dict['selection_function_was_evolved'] = self.selection_function_was_evolved
        info_dict['selection_function_eppsea_string'] = self.selection_function_eppsea_string

        info_dict['fitness_function_display_name'] = self.fitness_function_display_name
        info_dict['fitness_function_id'] = self.fitness_function_id

        return info_dict

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
        if len(self.fitvals) > 1 and (self.fitvals[0] == float('inf') or self.fitvals[-1] - self.fitvals[0] < 1e-12):
            res['tolfun'] = 1e-12
        if self.sigma * max(self.C.eigenvalues)**0.5 < 1e-11:
            # remark: max(D) >= max(diag(C))**0.5
            res['tolx'] = 1e-11
        return res

    def disp(self, verb_modulo=1):
        # overwrite parent disp function to reduce printouts
        pass

class CMAES_runner:
    def __init__(self, config, fitness_function, selection_function):
        self.config = config
        self.fitness_function = fitness_function
        self.selection_function = selection_function

    class Popi:
        def __init__(self):
            self.genome = None
            self.fitness = None

    def one_run(self, basic=False):
        start = self.fitness_function.random_genome()
        init_sigma = 0.5
        max_evals = self.config.getint('CMAES', 'maximum evaluations')
        population_size = self.config.getint('CMAES', 'population size')
        terminate_no_best_fitness_change = self.config.getboolean('CMAES', 'terminate on no improvement in best fitness')
        no_change_termination_generations = self.config.getint('CMAES', 'generations to termination for no improvement')


        es = ModifiedCMAES(start, init_sigma, maxfevals=max_evals, popsize=population_size)
        es.fitness_function = self.fitness_function

        self.fitness_function.start()

        generation = 0

        result = CMAES_Result()

        result.eval_counts = []
        result.fitnesses = dict()
        result.average_fitnesses = dict()
        result.best_fitnesses = dict()

        gens_since_best_fitness_improvement = 0
        best_fitness = math.inf
        term_conditions = None

        while not es.stop():
            es.sigma = min(es.sigma, 1e100)
            X = es.ask()  # get a list of sampled candidate solutions
            fitness_values = list(self.fitness_function.evaluate(x) for x in X)
            if basic:
                es.tell(X, fitness_values)
            else:
                population = []
                for x, fit in zip(X, fitness_values):
                    new_popi = self.Popi()
                    new_popi.genome = x
                    new_popi.fitness = -1 * fit  # eppsea assumes fitness maximization
                    population.append(new_popi)
                es.tell_pop(X, fitness_values, population, self.selection_function)  # update distribution parameters

            result.eval_counts.append(es.counteval)
            result.fitnesses[es.counteval] = fitness_values
            result.average_fitnesses[es.counteval] = statistics.mean(fitness_values)
            result.best_fitnesses[es.counteval] = max(fitness_values)

            if min(fitness_values) < best_fitness:
                best_fitness = min(fitness_values)
                gens_since_best_fitness_improvement = 0
            else:
                gens_since_best_fitness_improvement += 1
                if terminate_no_best_fitness_change and gens_since_best_fitness_improvement >= no_change_termination_generations:
                    term_conditions = {'fitness_stagnation': best_fitness}
                    break

            generation += 1

        es.disp(1)
        if term_conditions is None:
            term_conditions = es.stop()

        result.fitness_function_display_name = self.fitness_function.display_name
        result.fitness_function_id = self.fitness_function.id
        if basic:
            result.selection_function_was_evolved = False
            result.selection_function_display_name = 'Basic CMAES'
            result.selection_function_id = 'Basic_CMAES'
        else:
            result.selection_function_was_evolved = True
            result.selection_function_display_name = self.selection_function.display_name
            result.selection_function_id = self.selection_function.id
            result.selection_function_eppsea_string = self.selection_function.eppsea_selection_function.get_string()

        if 'ftarget' in term_conditions:
            result.termination_reason = 'target_fitness_hit'
        elif 'maxfevals' in term_conditions:
            result.termination_reason = 'maximum_evaluations_reached'
        elif 'tolfun' in term_conditions:
            result.termination_reason = 'fitness_convergence'
        elif 'tolx' in term_conditions:
            result.termination_reason = 'population_convergence'
        elif 'fitness_stagnation' in term_conditions:
            result.termination_reason = 'fitness_stagnation'

        result.final_best_fitness = max(es.best.f, self.fitness_function.evaluate(es.xmean))

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
        self.verbosity = config.getint('CMAES', 'logging verbosity')

        self.using_multiprocessing = config.getboolean('CMAES', 'use multiprocessing')

        self.test_generalization = config.getboolean('CMAES', 'test generalization')
        self.training_runs = config.getint('CMAES', 'training runs')
        self.testing_runs = config.getint('CMAES', 'testing runs')

        self.num_training_fitness_functions = config.getint('CMAES', 'num training fitness functions')
        self.num_testing_fitness_functions = config.getint('CMAES', 'num testing fitness functions')
        self.training_fitness_functions = None
        self.testing_fitness_functions = None

        self.basic_average_best_fitness = None
        self.basic_median_best_fitness = None

        if config.get('CMAES', 'eppsea fitness assignment method') == 'best fitness reached' or config.get('CMAES', 'eppsea fitness assignment method') == 'adaptive':
            self.eppsea_fitness_assignment_method = 'best_fitness_reached'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'proportion hitting target fitness':
            self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'evals to target fitness':
            self.eppsea_fitness_assignment_method = 'evals_to_target_fitness'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'proportion better than basic':
            self.eppsea_fitness_assignment_method = 'proportion_better_than_basic'
        elif config.get('CMAES', 'eppsea fitness assignment method') == 'proportion better than basic median':
            self.eppsea_fitness_assignment_method = 'proportion_better_than_basic_median'
        else:
            raise Exception('ERROR: eppsea fitness assignment method {0} not recognized!'.format(config.get('CMAES', 'eppsea fitness assignment method')))

        self.eppsea = None
        self.final_test_results = None

        self.prepare_fitness_functions(config)

    def log(self, message, verbosity):
        if self.verbosity >= verbosity:
            print(message)
            with open(self.log_file_location, 'a') as log_file:
                log_file.write(message + '\n')

    def prepare_fitness_functions(self, config):
        # loads the fitness functions to be used in the EAs
        fitness_function_config_path = config.get('CMAES', 'fitness function config path')
        fitness_function_config = configparser.ConfigParser()
        fitness_function_config.read(fitness_function_config_path)
        fitness_function_directory = config.get('CMAES', 'fitness function directory')

        if config.getboolean('CMAES', 'generate new fitness functions'):
            prepared_fitness_functions = ff.generate_coco_functions(fitness_function_config_path, True)
            os.makedirs(fitness_function_directory, exist_ok=True)
            for i, f in enumerate(prepared_fitness_functions):
                save_path = '{0}/{1}'.format(fitness_function_directory, i)
                ff.save(f, save_path)

        else:
            prepared_fitness_functions = []
            for fitness_function_path in sorted(os.listdir(fitness_function_directory)):
                full_fitness_function_path = '{0}/{1}'.format(fitness_function_directory, fitness_function_path)
                prepared_fitness_functions.append(ff.load(full_fitness_function_path))

        # sample a spread of the loaded fitness functions
        if len(prepared_fitness_functions) < self.num_training_fitness_functions:
            raise Exception('ERROR: Trying to load {0} training fitness functions, but only {1} are available'.format(self.num_training_fitness_functions, len(prepared_fitness_functions)))
        # take an even sampling of the training functions to prevent bias
        self.training_fitness_functions = []
        step_size =  len(prepared_fitness_functions) / self.num_training_fitness_functions
        i = 0
        for _ in range(self.num_training_fitness_functions):
            self.training_fitness_functions.append(prepared_fitness_functions[math.floor(i)])
            i += step_size
        if self.test_generalization:
            self.testing_fitness_functions = list(f for f in prepared_fitness_functions if f not in self.training_fitness_functions)
            if self.num_testing_fitness_functions == -1:
                self.num_testing_fitness_functions = len(self.testing_fitness_functions)
            else:
                self.testing_fitness_functions = self.testing_fitness_functions[:self.num_testing_fitness_functions]
        else:
            self.testing_fitness_functions = None

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

    def get_basic_cmaes_runners(self, fitness_functions):
        result = list()

        for fitness_function in fitness_functions:
            result.append(CMAES_runner(self.config, fitness_function, None))

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

        return all_run_results

    def run_basic_cmaes_runners(self, eas, is_testing):
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
            results = pool.map(run_basic_cmaes_runner, params)
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
                    all_run_results.append(run_basic_cmaes_runner(ea))

        return all_run_results

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

        self.log('Reporting results from EPPSEA generation {0}'.format(self.eppsea.gen_number), 2)
        for s in selection_functions:
            self.log('Results for EPPSEA member: {0}'.format(s.eppsea_selection_function.get_string()), 2)
            e_results = list(r for r in ea_results if r.selection_function_id == s.id)
            for f in fitness_functions:
                self.log('\tResults on fitness function {0} with id {1}'.format(f.display_name, f.id), 2)
                f_results = list(r for r in e_results if r.fitness_function_id == f.id)
                for r in f_results:
                    self.log('\t\tEvals: {0} | Final best fitness: {1} | Termination reason: {2}'.format(max(r.eval_counts), r.final_best_fitness, r.termination_reason), 2)
            self.log('\tFitness assigned to EPPSEA member: {0}'.format(s.eppsea_selection_function.fitness), 2)



    def assign_eppsea_fitness(self, selection_functions, ea_results):
        fitness_function_ids = set(r.fitness_function_id for r in ea_results)
        for s in selection_functions:
            s_results = list(r for r in ea_results if r.selection_function_id == s.id)
            if self.config.get('CMAES', 'eppsea fitness assignment method') == 'adaptive':
                if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                    if len(list(r for r in s_results if r.termination_reason == 'target_fitness_hit')) / len (s_results) >= 0.50:
                        self.log('At eval count {0}, eppsea fitness assignment changed to proportion_hitting_target_fitness'.format(self.eppsea.gp_evals), 1)
                        self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
                        for p in self.eppsea.population:
                            p.fitness = -math.inf
                        self.eppsea.gens_since_avg_fitness_improvement = 0
                        self.eppsea.gens_since_best_fitness_improvement = 0
                        self.eppsea.highest_average_fitness = -math.inf
                        self.eppsea.highest_best_fitness = -math.inf
                if self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                    if len(list(r for r in s_results if r.termination_reason == 'target_fitness_hit')) / len(s_results) >= .95:
                        self.log('At eval count {0}, eppsea fitness assignment changed to evals_to_target_fitness'.format(self.eppsea.gp_evals), 1)
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
            s_results = list(r for r in ea_results if r.selection_function_id == s.id)

            if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                # loop through all fitness functions to get average final best fitnesses
                average_final_best_fitnesses = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    final_best_fitnesses = (r.final_best_fitness for r in fitness_function_results)
                    average_final_best_fitnesses.append(statistics.mean(final_best_fitnesses))
                # assign fitness as -1 times the average of the average final best fitnesses
                s.eppsea_selection_function.fitness = -1 * statistics.mean(average_final_best_fitnesses)
                if s.eppsea_selection_function.fitness == float('-inf'):
                    s.eppsea_selection_function.fitness = -1e100

            elif self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                # assign the fitness as the proportion of runs that hit the target fitness
                s.eppsea_selection_function.fitness = len(list(r for r in s_results if r.termination_reason == 'target_fitness_hit')) / len (s_results)

            elif self.eppsea_fitness_assignment_method == 'evals_to_target_fitness':
                # loop through all fitness functions to get average evals to target fitness
                all_final_evals = []
                for fitness_function_id in fitness_function_ids:
                    final_evals = []
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    for r in fitness_function_results:
                        if r.termination_reason == 'target_fitness_hit':
                            final_evals.append(max(r.eval_counts))
                        else:
                            # for the runs where the target fitness was not hit, use an eval count equal to twice the maximum count
                            final_evals.append(2 * self.config.getint('CMAES', 'maximum evaluations'))

                    all_final_evals.append(statistics.mean(final_evals))
                # assign fitness as -1 * the average of final eval counts
                s.eppsea_selection_function.fitness = -1 * statistics.mean(all_final_evals)

            elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic':
                # assign the fitness as the proportion of runs that hit a fitness higher than cmaes did
                proportions_better = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness < self.basic_average_best_fitness[fitness_function_id])) / len(fitness_function_results)
                    proportions_better.append(proportion_better)
                # assign fitness as the average of the proportions better
                s.eppsea_selection_function.fitness = statistics.mean(proportions_better)

            elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic_median':
                # assign the fitness as the proportion of runs that hit a fitness higher than cmaes did
                proportions_better = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness < self.basic_median_best_fitness[fitness_function_id])) / len(fitness_function_results)
                    proportions_better.append(proportion_better)
                # assign fitness as the average of the proportions better
                s.eppsea_selection_function.fitness = statistics.mean(proportions_better)

            else:
                raise Exception('ERROR: fitness assignment method {0} not recognized by eppsea_basicEA'.format(self.eppsea_fitness_assignment_method))

    def run_final_tests(self, best_selection_function):
        if self.config.get('CMAES', 'test generalization'):
            fitness_functions = self.testing_fitness_functions + self.training_fitness_functions
        else:
            fitness_functions = self.training_fitness_functions

        eppsea_selection_function = SelectionFunction()
        eppsea_selection_function.generate_from_eppsea_individual(best_selection_function)

        cmaess = self.get_cmaes_runners(fitness_functions, [eppsea_selection_function])
        cmaes_results = self.run_cmaes_runners(cmaess, True)

        basic_cmaess = self.get_basic_cmaes_runners(fitness_functions)
        basic_cmaess_results = self.run_basic_cmaes_runners(basic_cmaess, True)

        return cmaes_results + basic_cmaess_results

    def save_final_results(self, final_results):
        file_path = self.results_directory + '/final_results'
        with open(file_path, 'wb') as file:
            pickle.dump(list(f.export() for f in final_results), file)

    def postprocess(self):
        postprocess_args = ['python3', 'post_process.py', self.results_directory, self.results_directory + '/final_results']
        output = subprocess.run(postprocess_args, stdout=subprocess.PIPE, universal_newlines=True).stdout
        return output

    def run_eppsea_cmaes(self):
        print('Now starting EPPSEA')
        start_time = time.time()

        eppsea_config = self.config.get('CMAES', 'base eppsea config path')
        eppsea = eppsea_base.Eppsea(eppsea_config)
        self.eppsea = eppsea

        if self.eppsea_fitness_assignment_method == 'proportion_better_than_basic':
            self.basic_average_best_fitness = dict()
            basic_cmaess = self.get_basic_cmaes_runners(self.training_fitness_functions)
            basic_cmaess_results = self.run_basic_cmaes_runners(basic_cmaess, True)
            for f in self.training_fitness_functions:
                f_results = list(r for r in basic_cmaess_results if r.fitness_function_id == f.id)
                self.basic_average_best_fitness[f.id] = statistics.mean(r.final_best_fitness for r in f_results)

        elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic_median':
            self.basic_median_best_fitness = dict()
            basic_cmaess = self.get_basic_cmaes_runners(self.training_fitness_functions)
            basic_cmaess_results = self.run_basic_cmaes_runners(basic_cmaess, True)
            for f in self.training_fitness_functions:
                f_results = list(r for r in basic_cmaess_results if r.fitness_function_id == f.id)
                self.basic_median_best_fitness[f.id] = statistics.median(r.final_best_fitness for r in f_results)

        eppsea.start_evolution()

        while not eppsea.evolution_finished:
            self.evaluate_eppsea_population(eppsea.new_population, False)
            eppsea.next_generation()

        best_selection_function = eppsea.final_best_member

        self.log('Running final tests', 1)
        self.log('Training fitness functions:', 2)
        for f in self.training_fitness_functions:
            self.log('Name: {0}, id: {1}'.format(f.display_name, f.id), 1)
        self.log('Testing fitness functions:', 2)
        for f in self.testing_fitness_functions:
            self.log('Name: {0}, id: {1}'.format(f.display_name, f.id), 1)
        self.final_test_results = self.run_final_tests(best_selection_function)
        self.save_final_results(self.final_test_results)

        self.log('Running Postprocessing', 1)
        postprocess_results = self.postprocess()
        self.log('Postprocess results:', 1)
        self.log(postprocess_results, 1)

        eppsea_base_results_path = eppsea.results_directory
        shutil.copytree(eppsea_base_results_path, self.results_directory + '/base')
        end_time = time.time() - start_time
        self.log('Total time elapsed: {0}'.format(end_time),1)


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = EppseaCMAES(config)
    shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))
    evaluator.run_eppsea_cmaes()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_paths = sys.argv[1:]

    for config_path in config_paths:
        main(config_path)