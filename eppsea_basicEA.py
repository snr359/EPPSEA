import math
import random
import statistics
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
import subprocess

import eppsea_base
import fitness_functions as ff

import numpy as np

def run_ea(ea):
    result = ea.one_run()
    return result

class EAResult:
    # a class for holding the results of a single run of an EA
    def __init__(self):
        self.eval_counts = None
        self.fitnesses = None
        self.average_fitnesses = None
        self.best_fitnesses = None
        self.final_best_fitness = None
        self.termination_reason = None

        self.selection_function = None
        self.selection_function_display_name = None
        self.selection_function_id = None
        self.selection_function_was_evolved = None
        self.selection_function_eppsea_string = None

        self.fitness_function = None
        self.fitness_function_display_name = None
        self.fitness_function_id = None

        self.id = None

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

class SelectionFunction:
    def __init__(self):
        self.eppsea_selection_function = None
        
        self.parent_selection_type = None
        self.parent_selection_tournament_k = None
        
        self.survival_selection_type = None
        self.survival_selection_tournament_k = None
        
        self.display_name = None

        self.id = None

    def __hash__(self):
        return hash(self.id)
        
    def assign_id(self):
        # assigns a random id to self. Every unique Selection Function should call this once
        self.id = '{0}_{1}_{2}'.format('SelectionFunction', str(id(self)), str(uuid.uuid4()))

    def generate_from_config(self, config):
        if config.getboolean('selection function', 'evolved'):
            file_path = config.get('selection function', 'file path (for evolved selection)')
            with open(file_path, 'rb') as file:
                self.eppsea_selection_function = pickle.load(file)
            
        else:
            self.parent_selection_type = config.get('selection function', 'parent selection type')
            if self.parent_selection_type in ('k_tournament_replacement', 'k_tournament_no_replacement'):
                self.parent_selection_tournament_k = config.getint('selection function', 'parent selection tournament k')
    
            self.survival_selection_type = config.get('selection function', 'survival selection type')
            if self.survival_selection_type in ('k_tournament_replacement', 'k_tournament_no_replacement'):
                self.survival_selection_tournament_k = config.getint('selection function', 'survival selection tournament k')
            
        self.display_name = config.get('selection function', 'display name')
        
        self.assign_id()

    def generate_from_eppsea_individual(self, eppsea_selection_function):
        self.parent_selection_type = 'eppsea_seleciton_function'
        self.parent_selection_tournament_k = None

        self.survival_selection_type = 'eppsea_seleciton_function'
        self.survival_selection_tournament_k = None

        self.display_name = 'Evolved Selection Function'
        self.eppsea_selection_function = eppsea_selection_function
        self.assign_id()
        
    def parent_selection(self, population, n, generation_number, minimizing):
        if minimizing:
            for p in population:
                p.fitness *= -1

        if self.eppsea_selection_function is not None:
            selected =  self.eppsea_selection_function.select(population, n, 0, generation_number)
        else:
            selected =  self.basic_selection(population, n, self.parent_selection_type, self.parent_selection_tournament_k)

        if minimizing:
            for p in population:
                p.fitness *= -1

        return selected
        
    def survival_selection(self, population, n, generation_number, minimizing):
        if minimizing:
            for p in population:
                p.fitness *= -1

        if self.eppsea_selection_function is not None:
            selected = self.eppsea_selection_function.select(population, n, 1, generation_number)
        else:
            selected =  self.basic_selection(population, n, self.survival_selection_type, self.survival_selection_tournament_k)

        if minimizing:
            for p in population:
                p.fitness *= -1

        return selected

    def basic_selection(self, population, n, selection_type, tournament_k):
        if selection_type == 'truncation':
            return sorted(population, key=lambda x: x.fitness, reverse=True)[:n]

        elif selection_type == 'fitness_rank_replacement':
            selected = []
            sorted_population = sorted(population, key=lambda p: p.fitness, reverse=True)
            ranks = list(range(len(sorted_population), 0, -1))
            sum_ranks = sum(ranks)
            for _ in range(n):
                r = random.randint(0, sum_ranks)
                i = 0
                while r > ranks[i]:
                    r -= ranks[i]
                    i += 1
                selected.append(sorted_population[i])
            return selected
        
        elif selection_type == 'fitness_rank_no_replacement':
            selected = []
            sorted_population = sorted(population, key=lambda p: p.fitness, reverse=True)
            ranks = list(range(len(sorted_population), 0, -1))
            for _ in range(n):
                sum_ranks = sum(ranks)
                r = random.randint(0, sum_ranks)
                i = 0
                while r > ranks[i]:
                    r -= ranks[i]
                    i += 1
                selected.append(sorted_population[i])
                sorted_population.pop(i)
                ranks.pop(i)
            return selected

        elif selection_type == 'fitness_proportional_replacement':
            selected = []
            min_fitness = min(p.fitness for p in population)
            if min_fitness < 0:
                selection_chances = [p.fitness - min_fitness for p in population]
            else:
                selection_chances = [p.fitness for p in population]
            sum_selection_chances = sum(selection_chances)
            for _ in range(n):
                r = random.uniform(0, sum_selection_chances)
                i = 0
                while r > selection_chances[i]:
                    r -= selection_chances[i]
                    i += 1
                selected.append(population[i])
            return selected
        
        elif selection_type == 'fitness_proportional_no_replacement':
            selected = []
            population_copy = list(population)
            min_fitness = min(p.fitness for p in population_copy)
            if min_fitness < 0:
                selection_chances = [p.fitness - min_fitness for p in population_copy]
            else:
                selection_chances = [p.fitness for p in population_copy]
            for _ in range(n):
                sum_selection_chances = sum(selection_chances)
                r = random.uniform(0, sum_selection_chances)
                i = 0
                while r > selection_chances[i]:
                    r -= selection_chances[i]
                    i += 1
                selected.append(population_copy[i])
                population_copy.pop(i)
                selection_chances.pop(i)
            return selected

        elif selection_type == 'k_tournament_replacement':
            selected = []
            for _ in range(n):
                tournament = random.sample(population, tournament_k)
                winner = max(tournament, key=lambda p: p.fitness)
                selected.append(winner)
            return selected

        elif selection_type == 'k_tournament_no_replacement':
            selected = []
            population_copy = list(population)
            for _ in range(n):
                tournament = random.sample(population_copy, tournament_k)
                winner = max(tournament, key=lambda p: p.fitness)
                selected.append(winner)
                population_copy.remove(winner)
            return selected

        elif selection_type == 'random':
            selected = []
            for _ in range(n):
                selected.append(random.choice(population))
            return selected

        elif selection_type == 'stochastic_universal_sampling':
            # normalize the weights, if necessary
            normalized_weights = list(p.fitness for p in population)
            min_weight = min(normalized_weights)
            if min_weight < 0:
                for i in range(len(normalized_weights)):
                    normalized_weights[i] -= min_weight

            # if the sum weight is 0 or inf, set all normalized weights to 1
            sum_weight = sum(normalized_weights)
            if sum_weight == 0 or sum_weight == math.inf:
                normalized_weights = [1] * len(population)

            # build a list of the indices and cumulative selection weights
            indices_and_weights = []
            cum_weight = 0
            for i in range(len(population)):
                cum_weight += normalized_weights[i]
                indices_and_weights.append((i, cum_weight))
            sum_weight = cum_weight

            # calculate interval length
            interval_length = sum_weight / n

            # calculate initial interval offset
            offset = random.uniform(0, interval_length)

            # select population members at interval points
            selected = []
            for i, w in indices_and_weights:
                while offset < w:
                    selected.append(population[i])
                    offset += interval_length

            return selected

        else:
            print('BASIC SELECTION {0} NOT FOUND'.format(selection_type))


class EA:
    def __init__(self, config, fitness_function, selection_function):
        self.mu = config.getint('EA', 'population size')
        self.lam = config.getint('EA', 'offspring size')

        self.mutation_rate = config.getfloat('EA', 'mutation rate')
        self.max_evals = config.getint('EA', 'maximum evaluations')
        self.terminate_on_fitness_convergence = config.getboolean('EA', 'terminate on fitness convergence')
        self.generations_to_convergence = config.getint('EA', 'generations to convergence')
        self.minimize_fitness_function = config.getboolean('EA', 'minimize fitness function')
        self.terminate_at_target_fitness = config.getboolean('EA', 'terminate at target fitness')
        self.use_custom_target_fitness = config.getboolean('EA', 'use custom target fitness')
        self.target_fitness = config.getfloat('EA', 'target fitness')
        self.terminate_on_population_convergence = config.getboolean('EA', 'terminate on population convergence')
        self.population_convergence_threshold = config.getfloat('EA', 'population convergence threshold')

        self.fitness_function = fitness_function
        self.selection_function = selection_function

        if self.selection_function.eppsea_selection_function is not None:
            self.selection_function_eppsea_string = self.selection_function.eppsea_selection_function.get_string()

    class Popi:
        def __init__(self, other=None):
            if other is None:
                self.genome = None
                self.genome_type = None
                self.genome_length = None
                self.fitness = None
            else:
                self.genome = np.copy(other.genome)
                self.genome_type = other.genome_type
                self.genome_length = other.genome_length

        def randomize(self, n, max_range, genome_type):
            self.genome_length = n
            if genome_type == 'bool':
                self.genome_type = 'bool'
                self.genome = np.random.random(n) > 0.5
            elif genome_type == 'float':
                self.genome_type = 'float'
                self.genome = np.random.uniform(-max_range, max_range, n)

        def mutate_gene(self, gene):
            if self.genome_type == 'bool':
                self.genome[gene] = not self.genome[gene]

            elif self.genome_type == 'float':
                self.genome[gene] += random.uniform(-1, 1)

        def mutate_one(self):
            gene = random.randrange(self.genome_length)
            self.mutate_gene(gene)

        def mutate_all(self):
            for i in range(self.genome_length):
                self.mutate_gene(i)

        def recombine(self, parent2):
            new_child = EA.Popi(self)
            if self.genome_type == 'bool':
                for i in range(self.genome_length):
                    if random.random() > 0.5:
                        new_child.genome[i] = parent2.genome[i]
            elif self.genome_type == 'float':
                a = np.random.random(self.genome_length)
                new_child.genome = a*self.genome + (1-a)*parent2.genome

            return new_child

    def evaluate_child(self, popi, fitness_function):
        popi.fitness = fitness_function.evaluate(popi.genome)

    def one_run(self):
        # does a single ea run, and returns an EAResult
        generation_number = 0
        termination_reason = None
        self.fitness_function.start()

        # initialize the population
        population = list()
        for _ in range(self.mu):
            new_child = self.Popi()
            new_child.randomize(self.fitness_function.genome_length, self.fitness_function.max_initial_range, self.fitness_function.genome_type)
            new_child.birth_generation = generation_number
            self.evaluate_child(new_child, self.fitness_function)
            population.append(new_child)

        # record the initial evaluation count
        evals = self.mu

        # set up data holders and store information
        eval_counts = [evals]

        fitnesses = dict()
        fitnesses[evals] = list(p.fitness for p in population)

        average_fitnesses = dict()
        average_fitnesses[evals] = statistics.mean(p.fitness for p in population)

        best_fitnesses = dict()
        best_fitnesses[evals] = max(p.fitness for p in population)

        generations_since_best_fitness_improvement = 0
        previous_best_fitness = -math.inf

        generation_number = 1

        # main ea loop
        while evals <= self.max_evals:
            children = []

            # select parents
            num_parents = self.lam*2
            all_parents = self.selection_function.parent_selection(population, num_parents, generation_number, self.minimize_fitness_function)
            # pair up parents and recombine them
            for i in range(0, len(all_parents), 2):
                parent1 = all_parents[i]
                parent2 = all_parents[i + 1]

                new_child = parent1.recombine(parent2)
                new_child.birth_generation = generation_number
                # chance to mutate each new gene of the child
                for j in range(new_child.genome_length):
                    if random.random() < self.mutation_rate:
                        new_child.mutate_gene(j)

                # evaluate the new child
                self.evaluate_child(new_child, self.fitness_function)

                children.append(new_child)

                evals += 1

            # add all the new children to the population
            population.extend(children)

            # perform survival selection
            survivors = self.selection_function.survival_selection(population, self.mu, generation_number, self.minimize_fitness_function)
            population = survivors

            # record fitness values
            eval_counts.append(evals)

            fitnesses[evals] = list(p.fitness for p in population)

            average_fitness = statistics.mean(p.fitness for p in population)
            average_fitnesses[evals] = average_fitness

            best_fitness = max(p.fitness for p in population)
            best_fitnesses[evals] = best_fitness

            generation_number += 1

            # check for termination conditions
            if self.terminate_at_target_fitness:
                if self.use_custom_target_fitness:
                    if any(p.fitness >= self.target_fitness for p in population):
                        termination_reason = 'target_fitness_hit'
                        break
                else:
                    if self.fitness_function.fitness_target_hit():
                        termination_reason = 'target_fitness_hit'
                        break

            if self.terminate_on_population_convergence:
                genomes = np.stack(p.genome for p in population)
                num_unique_genomes = len(np.unique(genomes, axis=0))
                if num_unique_genomes < self.population_convergence_threshold * self.mu:
                    termination_reason = 'population_convergence'
                    break

            if best_fitness > previous_best_fitness:
                previous_best_fitness = best_fitness
                generations_since_best_fitness_improvement = 0
            else:
                generations_since_best_fitness_improvement += 1
                if self.terminate_on_fitness_convergence and generations_since_best_fitness_improvement >= self.generations_to_convergence:
                    termination_reason = 'fitness_convergence'
                    break

        if termination_reason is None:
            termination_reason = 'maximum_evaluations_reached'

        self.fitness_function.finish()

        # store results in an EAResults object
        run_results = EAResult()
        run_results.eval_counts = list(average_fitnesses.keys())
        run_results.final_average_fitness = statistics.mean(p.fitness for p in population)
        run_results.final_best_fitness = max(p.fitness for p in population)
        run_results.final_fitness_std_dev = statistics.stdev(p.fitness for p in population)
        run_results.average_fitnesses = average_fitnesses
        run_results.best_fitnesses = best_fitnesses
        run_results.termination_reason = termination_reason

        run_results.eval_counts = eval_counts

        run_results.fitnesses = fitnesses

        run_results.selection_function = self.selection_function
        run_results.selection_function_display_name = self.selection_function.display_name
        run_results.selection_function_id = self.selection_function.id

        if self.selection_function.eppsea_selection_function is not None:
            run_results.selection_function_was_evolved = True

        run_results.fitness_function = self.fitness_function
        run_results.fitness_function_display_name = self.fitness_function.display_name
        run_results.fitness_function_id = self.fitness_function.id

        return run_results


class EppseaBasicEA:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool',
        'mk_landscape': 'bool'
    }

    def __init__(self, config):
        self.config = config

        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_basicEA_" + str(present_time)

        self.results_directory = 'results/eppsea_basicEA/{0}'.format(experiment_name)
        os.makedirs(self.results_directory, exist_ok=True)

        self.log_file_location = '{0}/log.txt'.format(self.results_directory)

        self.using_multiprocessing = config.getboolean('EA', 'use multiprocessing')

        self.test_generalization = config.getboolean('EA', 'test generalization')
        self.training_runs = config.getint('EA', 'training runs')
        self.testing_runs = config.getint('EA', 'testing runs')
        self.use_multiobjective_ea = config.getboolean('EA', 'use multiobjective ea')
        self.minimize_fitness_function = config.getboolean('EA', 'minimize fitness function')

        self.use_preliminary_training_runs = config.getboolean('EA', 'use preliminary training runs')
        self.preliminary_training_runs = config.getint('EA', 'preliminary training runs')
        self.preliminary_fitness_threshold = config.getfloat('EA', 'preliminary fitness threshold')

        # if we are using adaptive fitness assignment, start fitness assignment method as best_fitness_reached
        if config.get('EA', 'eppsea fitness assignment method') == 'best fitness reached' or config.get('EA', 'eppsea fitness assignment method') == 'adaptive':
            self.eppsea_fitness_assignment_method = 'best_fitness_reached'
        elif config.get('EA', 'eppsea fitness assignment method') == 'proportion hitting target fitness':
            self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
        elif config.get('EA', 'eppsea fitness assignment method') == 'evals to target fitness':
            self.eppsea_fitness_assignment_method = 'evals_to_target_fitness'
        elif config.get('EA', 'eppsea fitness assignment method') == 'proportion better than basic':
            self.eppsea_fitness_assignment_method = 'proportion_better_than_basic'
        elif config.get('EA', 'eppsea fitness assignment method') == 'proportion better than basic median':
            self.eppsea_fitness_assignment_method = 'proportion_better_than_basic_median'
        else:
            raise Exception('ERROR: eppsea fitness assignment method {0} not recognized!'.format(config.get('EA', 'eppsea fitness assignment method')))

        self.num_training_fitness_functions = config.getint('EA', 'num training fitness functions')
        self.num_testing_fitness_functions = config.getint('EA', 'num testing fitness functions')
        self.training_fitness_functions = None
        self.testing_fitness_functions = None

        self.basic_average_best_fitness = None
        self.basic_average_best_evals = None
        self.basic_median_best_fitness = None
        self.basic_median_best_evals = None

        self.measuring_against_basic_evals = None

        self.eppsea = None

        self.prepare_fitness_functions(config)

        self.basic_selection_functions = None
        self.prepare_basic_selection_functions(config)

        self.basic_results = None

    def prepare_fitness_functions(self, config):
        # loads the fitness functions to be used in the EAs
        fitness_function_config_path = config.get('EA', 'fitness function config path')
        fitness_function_config = configparser.ConfigParser()
        fitness_function_config.read(fitness_function_config_path)
        fitness_function_directory = config.get('EA', 'fitness function directory')

        if config.getboolean('EA', 'generate new fitness functions'):
            num_fitness_functions = self.num_training_fitness_functions + self.num_testing_fitness_functions # ignored for coco functions
            prepared_fitness_functions = ff.generate_fitness_functions(fitness_function_config_path, True, num_fitness_functions)
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
            
    def prepare_basic_selection_functions(self, config):
        self.basic_selection_functions = []
        selection_configs = config.items('basic selection function configs')
        for _, selection_config_path in selection_configs:
            selection_config = configparser.ConfigParser()
            selection_config.read(selection_config_path)
            basic_selection_function = SelectionFunction()
            basic_selection_function.generate_from_config(selection_config)
            self.basic_selection_functions.append(basic_selection_function)

    def log(self, message):
        print(message)
        with open(self.log_file_location, 'a') as log_file:
            log_file.write(message + '\n')

    def get_eas(self, fitness_functions, selection_functions):
        # this prepares and returns a list of EAs for the provided selection functions
        # selection_functions is a list of tuples of the form (selection_function_name, selection_function),
        # where, if the selection_function_name is 'eppsea_selection_function', then selection_function is
        # expected to be an eppsea selection function

        result = list()

        for selection_function in selection_functions:
            for fitness_function in fitness_functions:
                result.append(EA(self.config, fitness_function, selection_function))

        return result

    def run_eas(self, eas, is_testing, is_preliminary):
        # runs each of the eas for the configured number of runs, returning one result_holder for each ea
        all_run_results = []

        if is_testing:
            runs = self.testing_runs
        else:
            if is_preliminary:
                runs = self.preliminary_training_runs
            else:
                runs = self.training_runs

        if self.using_multiprocessing:
            # setup parameters for multiprocessing
            params = []
            for ea in eas:
                params.extend([ea]*runs)
            # run all runs
            pool = multiprocessing.Pool()
            results = pool.map(run_ea, params)
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
                    all_run_results.append(run_ea(ea))

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

        if self.use_preliminary_training_runs:
            eas = self.get_eas(fitness_functions, selection_functions)
            preliminary_results =  self.run_eas(eas, False, True)
            good_selection_functions = []
            for s in selection_functions:
                s_results = list(r for r in preliminary_results if r.selection_function_id == s.id)
                if any(r.termination_reason == 'hit_target_fitness' or r.final_best_fitness < self.preliminary_fitness_threshold for r in s_results):
                    good_selection_functions.append(s)
            if good_selection_functions:
                eas = self.get_eas(fitness_functions, good_selection_functions)
                good_selection_function_results = self.run_eas(eas, False, False)
            else:
                good_selection_function_results = []
            ea_results = preliminary_results + good_selection_function_results
        else:
            eas = self.get_eas(fitness_functions, selection_functions)
            ea_results = self.run_eas(eas, False, False)
        self.assign_eppsea_fitness(selection_functions, ea_results)

    def assign_eppsea_fitness(self, selection_functions, ea_results):
        # takes an EAResultsCollection and uses it to assign fitness values to the eppsea_population

        # if we are using adaptive fitness assignment, then when Eppsea_basicEA starts, eppsea fitness is assigned by average best fitness reached on the bottom-level EA
        # if we are currently assigning eppsea fitness by looking at average best fitness reached, and at least 50% of
        # the runs of any one selection function are reaching the fitness goal, switch to assigning eppsea fitness by proportion of time
        # the fitness goal is found
        # if we are currently assigning by that, and at least 95% of the runs runs of any one selection function are reaching the fitness goal, switch
        # to assigning by number of evals needed to reach the fitness goal
        # in the case of either switch, all population members' fitnesses become -infinity before fitness assignment
        # this effectively sets the fitness of all old eppsea population members to -infinity
        # the counters for average/best fitness change at the eppsea level are also manually reset, since this counts as a fitness improvement
        fitness_function_ids = set(r.fitness_function_id for r in ea_results)

        for s in selection_functions:
            s_results = list(r for r in ea_results if r.selection_function_id == s.id)
            if self.config.get('EA', 'eppsea fitness assignment method') == 'adaptive':
                if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                    if len(list(r for r in s_results if r.termination_reason == 'target_fitness_hit')) / len (s_results) >= 0.5:
                        self.log('At eval count {0}, eppsea fitness assignment changed to proportion_hitting_target_fitness'.format(self.eppsea.gp_evals))
                        self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
                        for p in self.eppsea.population:
                            p.fitness = -math.inf
                        self.eppsea.gens_since_avg_fitness_improvement = 0
                        self.eppsea.gens_since_best_fitness_improvement = 0
                        self.eppsea.highest_average_fitness = -math.inf
                        self.eppsea.highest_best_fitness = -math.inf
                if self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                    if len(list(r for r in s_results if r.termination_reason == 'target_fitness_hit')) / len(s_results) >= .95:
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
            s_results = list(r for r in ea_results if r.selection_function_id == s.id)

            if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
                # loop through all fitness functions to get average final best fitnesses
                average_final_best_fitnesses = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    final_best_fitnesses = (r.final_best_fitness for r in fitness_function_results)
                    average_final_best_fitnesses.append(statistics.mean(final_best_fitnesses))
                # assign fitness as the average of the average final best fitnesses or, if multiobjective ea is on, the list of average final best fitnesses
                if self.use_multiobjective_ea:
                    s.eppsea_selection_function.mo_fitnesses = average_final_best_fitnesses
                    if self.minimize_fitness_function:
                        for i,f in enumerate(s.eppsea_selection_function.mo_fitnesses):
                            s.eppsea_selection_function.mo_fitnesses[i] *= -1

                else:
                    s.eppsea_selection_function.fitness = statistics.mean(average_final_best_fitnesses)
                    if self.minimize_fitness_function:
                        s.eppsea_selection_function.fitness *= -1


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
                            final_evals.append(2 * self.config.getint('EA', 'maximum evaluations'))

                    all_final_evals.append(statistics.mean(final_evals))
                # assign fitness as -1 * the average of final eval counts
                s.eppsea_selection_function.fitness = -1 * statistics.mean(all_final_evals)

            elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic':
                # assign the fitness as the proportion of runs that hit a fitness higher than cmaes did
                proportions_better = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    if self.measuring_against_basic_evals:
                        proportion_better = len(list(r for r in fitness_function_results if r.termination_reason == 'target_fitness_hit' and (r.eval_counts) < self.basic_average_best_evals[fitness_function_id])) / len(fitness_function_results)
                        proportions_better.append(proportion_better)
                    else:
                        if self.minimize_fitness_function:
                            proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness < self.basic_average_best_fitness[fitness_function_id])) / len(fitness_function_results)
                            proportions_better.append(proportion_better)
                        else:
                            proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness > self.basic_average_best_fitness[fitness_function_id])) / len(fitness_function_results)
                            proportions_better.append(proportion_better)
                # assign fitness as the average of the proportions better
                s.eppsea_selection_function.fitness = statistics.mean(proportions_better)

            elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic_median':
                # assign the fitness as the proportion of runs that hit a fitness higher than cmaes did
                proportions_better = []
                for fitness_function_id in fitness_function_ids:
                    fitness_function_results = list(r for r in s_results if r.fitness_function_id == fitness_function_id)
                    if self.measuring_against_basic_evals:
                        proportion_better = len(list(r for r in fitness_function_results if r.termination_reason == 'target_fitness_hit' and (r.eval_counts) < self.basic_median_best_evals[fitness_function_id])) / len(fitness_function_results)
                        proportions_better.append(proportion_better)
                    else:
                        if self.minimize_fitness_function:
                            proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness < self.basic_median_best_fitness[fitness_function_id])) / len(fitness_function_results)
                            proportions_better.append(proportion_better)
                        else:
                            proportion_better = len(list(r for r in fitness_function_results if r.final_best_fitness > self.basic_median_best_fitness[fitness_function_id])) / len(fitness_function_results)
                            proportions_better.append(proportion_better)
                # assign fitness as the average of the proportions better
                s.eppsea_selection_function.fitness = statistics.mean(proportions_better)

            else:
                raise Exception('ERROR: fitness assignment method {0} not recognized by eppsea_basicEA'.format(self.eppsea_fitness_assignment_method))

    def test_against_basic_selection(self, eppsea_individuals):

        selection_functions = self.basic_selection_functions
        if self.config.getint('EA', 'offspring size') > self.config.getint('EA', 'population size') * 2:
            for s in list(selection_functions):
                if s.parent_selection_type == 'truncation':
                    selection_functions.remove(s)

        for i, eppsea_individual in enumerate(eppsea_individuals):
            eppsea_selection_function = SelectionFunction()
            eppsea_selection_function.generate_from_eppsea_individual(eppsea_individual)
            eppsea_selection_function.display_name += ' {0}'.format(i)
            selection_functions.append(eppsea_selection_function)

        if self.test_generalization:
            fitness_functions = self.testing_fitness_functions + self.training_fitness_functions
        else:
            fitness_functions = self.training_fitness_functions

        eas = self.get_eas(fitness_functions, selection_functions)

        ea_results = self.run_eas(eas, True, False)

        return ea_results

    def postprocess(self):
        postprocess_args = ['python3', 'post_process.py', self.results_directory, self.results_directory + '/final_results']
        output = subprocess.run(postprocess_args, stdout=subprocess.PIPE, universal_newlines=True).stdout
        return output


    def convert_to_eppsea_selection(self, selection_functions):
        # converts a list of SelectionFunction objects to eppsea_base.EppseaSelectionFunction objects
        results = []
        for s in selection_functions:
            # if this selection function is already based on an eppsea selection function, just use it
            if s.eppsea_selection_function is not None:
                results.append(s.eppsea_selection_function)
            # otherwise, build a new eppsea selection function with the same behavior
            else:
                result = eppsea_base.EppseaSelectionFunction()
                result.number_of_selectors = 2
                result.selection_parameters = self.eppsea.selection_parameters

                # build the parent selection tree
                parent_selection = eppsea_base.GPTree()
                
                parent_selection.selection_parameters = result.selection_parameters[0]

                parent_selection.root = eppsea_base.GPNode(parent_selection.selection_parameters)
                
                if s.parent_selection_type == 'truncation':
                    parent_selection.root.operation = 'fitness'
                    
                    parent_selection.selection_type = 'truncation'
                    parent_selection.tournament_size = 0
                
                elif s.parent_selection_type == 'fitness_rank_replacement':
                    parent_selection.root.operation = 'fitness_rank'

                    parent_selection.selection_type = 'proportional_replacement'
                    parent_selection.tournament_size = 0
                    
                elif s.parent_selection_type == 'fitness_rank_no_replacement':
                    parent_selection.root.operation = 'fitness_rank'

                    parent_selection.selection_type = 'proportional_no_replacement'
                    parent_selection.tournament_size = 0
                    
                elif s.parent_selection_type == 'fitness_proportional_replacement':
                    parent_selection.root.operation = 'fitness'

                    parent_selection.selection_type = 'proportional_replacement'
                    parent_selection.tournament_size = 0
                    
                elif s.parent_selection_type == 'fitness_proportional_no_replacement':
                    parent_selection.root.operation = 'fitness'

                    parent_selection.selection_type = 'proportional_no_replacement'
                    parent_selection.tournament_size = 0
                
                elif s.parent_selection_type == 'k_tournament_replacement':
                    parent_selection.root.operation = 'fitness'

                    parent_selection.selection_type = 'tournament_replacement'
                    parent_selection.tournament_size = s.parent_selection_tournament_k
                    
                elif s.parent_selection_type == 'k_tournament_no_replacement':
                    parent_selection.root.operation = 'fitness'

                    parent_selection.selection_type = 'tournament_no_replacement'
                    parent_selection.tournament_size = s.parent_selection_tournament_k
                
                elif s.parent_selection_type == 'random':
                    parent_selection.root.operation = 'constant'

                    parent_selection.selection_type = 'proportional'
                    parent_selection.tournament_size = 0
                    parent_selection.data = 1
                    
                elif s.parent_selection_type == 'stochastic_universal_sampling':
                    parent_selection.root.operation = 'fitness'

                    parent_selection.selection_type = 'stochastic_universal_sampling'
                    parent_selection.tournament_size = 0

                # build the survival selection tree
                survival_selection = eppsea_base.GPTree()

                survival_selection.selection_parameters = result.selection_parameters[1]

                survival_selection.root = eppsea_base.GPNode(survival_selection.selection_parameters)

                if s.survival_selection_type == 'truncation':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'truncation'
                    survival_selection.tournament_size = 0

                elif s.survival_selection_type == 'fitness_rank_replacement':
                    survival_selection.root.operation = 'fitness_rank'

                    survival_selection.selection_type = 'proportional_replacement'
                    survival_selection.tournament_size = 0

                elif s.survival_selection_type == 'fitness_rank_no_replacement':
                    survival_selection.root.operation = 'fitness_rank'

                    survival_selection.selection_type = 'proportional_no_replacement'
                    survival_selection.tournament_size = 0

                elif s.survival_selection_type == 'fitness_proportional_replacement':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'proportional_replacement'
                    survival_selection.tournament_size = 0

                elif s.survival_selection_type == 'fitness_proportional_no_replacement':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'proportional_no_replacement'
                    survival_selection.tournament_size = 0

                elif s.survival_selection_type == 'k_tournament_replacement':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'tournament_replacement'
                    survival_selection.tournament_size = s.survival_selection_tournament_k

                elif s.survival_selection_type == 'k_tournament_no_replacement':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'tournament_no_replacement'
                    survival_selection.tournament_size = s.survival_selection_tournament_k

                elif s.survival_selection_type == 'random':
                    survival_selection.root.operation = 'constant'

                    survival_selection.selection_type = 'proportional'
                    survival_selection.tournament_size = 0
                    survival_selection.data = 1

                elif s.survival_selection_type == 'stochastic_universal_sampling':
                    survival_selection.root.operation = 'fitness'

                    survival_selection.selection_type = 'stochastic_universal_sampling'
                    survival_selection.tournament_size = 0

                # assign the parent and survival selection trees to the selection function
                result.gp_trees = [parent_selection, survival_selection]

                results.append(result)

        return results

    def save_final_results(self, final_results):
        file_path = self.results_directory + '/final_results'
        with open(file_path, 'wb') as file:
            pickle.dump(list(f.export() for f in final_results), file, protocol=4)

    def run_eppsea_basicea(self):
        print('Now starting EPPSEA')
        start_time = time.time()

        eppsea_config = self.config.get('EA', 'base eppsea config path')
        eppsea = eppsea_base.Eppsea(eppsea_config)
        self.eppsea = eppsea

        if self.eppsea_fitness_assignment_method == 'proportion_better_than_basic':
            self.basic_average_best_fitness = dict()
            self.basic_average_best_evals = dict()
            self.measuring_against_basic_evals = False
            basic_eas = self.get_eas(self.training_fitness_functions, self.basic_selection_functions)
            basic_ea_results = self.run_eas(basic_eas, True, False)
            for f in self.training_fitness_functions:
                f_results = list(r for r in basic_ea_results if r.fitness_function_id == f.id)
                self.basic_average_best_fitness[f.id] = statistics.mean(r.final_best_fitness for r in f_results)

                if len(list(r for r in f_results if r.termination_reason == 'target_fitness_reached')) / len(f_results) >= .95:
                    self.measuring_against_basic_evals = True
                evals = []
                for r in f_results:
                    if r.termination_reason == 'target_fitness_reached':
                        evals.append(max(r.eval_counts))
                    else:
                        evals.append(2 * self.config.getint('EA', 'maximum evaluations'))
                self.basic_average_best_evals = statistics.mean(evals)

        elif self.eppsea_fitness_assignment_method == 'proportion_better_than_basic_median':
            self.basic_median_best_fitness = dict()
            self.basic_median_best_evals = dict()
            self.measuring_against_basic_evals = False
            basic_eas = self.get_eas(self.training_fitness_functions, self.basic_selection_functions)
            basic_ea_results = self.run_eas(basic_eas, True, False)
            for f in self.training_fitness_functions:
                f_results = list(r for r in basic_ea_results if r.fitness_function_id == f.id)
                self.basic_median_best_fitness[f.id] = statistics.median(r.final_best_fitness for r in f_results)

                if len(list(r for r in f_results if r.termination_reason == 'target_fitness_reached')) / len(f_results) >= 95:
                    self.measuring_against_basic_evals = True
                evals = []
                for r in f_results:
                    if r.termination_reason == 'target_fitness_reached':
                        evals.append(max(r.eval_counts))
                    else:
                        evals.append(2 * self.config.getint('EA', 'maximum evaluations'))
                self.basic_median_best_evals = statistics.median(evals)

        initial_members = self.convert_to_eppsea_selection(self.basic_selection_functions)
        eppsea.initial_population.extend(initial_members)

        eppsea.start_evolution()

        while not eppsea.evolution_finished:
            self.evaluate_eppsea_population(eppsea.new_population, False)
            eppsea.next_generation()

        if self.use_multiobjective_ea:
            best_selection_functions = eppsea.final_best_members
        else:
            best_selection_functions = [eppsea.final_best_member]
        print('Running final tests')

        self.final_test_results = self.test_against_basic_selection(best_selection_functions)
        self.save_final_results(self.final_test_results)

        self.log('Running Postprocessing')
        postprocess_results = self.postprocess()
        self.log('Postprocess results:')
        self.log(postprocess_results)

        eppsea_base_results_path = eppsea.results_directory
        shutil.copytree(eppsea_base_results_path, self.results_directory + '/base')
        end_time = time.time() - start_time
        self.log('Total time elapsed: {0}'.format(end_time))

def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = EppseaBasicEA(config)
    shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))
    evaluator.run_eppsea_basicea()

    # pickle the entire eppsea_basicEA object, and separately the base selection function found and a config file for it, and the final test results
    evaluator_pickle_path = '{0}/EppseaBasicEA'.format(evaluator.results_directory)
    with open(evaluator_pickle_path, 'wb') as pickle_file:
        pickle.dump(evaluator, pickle_file, protocol=4)
    
    if not evaluator.use_multiobjective_ea:
        selection_function_pickle_path = '{0}/EvolvedSelectionFunction'.format(evaluator.results_directory)
        with open(selection_function_pickle_path, 'wb') as pickle_file:
            pickle.dump(evaluator.eppsea.final_best_member, pickle_file, protocol=4)
        selection_function_config_path = '{0}/EvolvedSelectionFunction.cfg'.format(evaluator.results_directory)
        selection_function_config = configparser.ConfigParser()
        selection_function_config.add_section('selection function')
        selection_function_config['selection function']['evolved'] = 'True'
        selection_function_config['selection function']['file path (for evolved selection)'] = selection_function_pickle_path
        selection_function_config['selection function']['display name'] = 'Evolved Selection Function'
        selection_function_config['selection function']['parent selection type'] = 'none'
        selection_function_config['selection function']['parent selection tournament k'] = '0'
        selection_function_config['selection function']['survival selection type'] = 'none'
        selection_function_config['selection function']['survival selection tournament k'] = '0'
        with open(selection_function_config_path, 'w') as selection_function_config_file:
            selection_function_config.write(selection_function_config_file)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    main(config_path)
