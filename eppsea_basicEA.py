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

import eppsea_base

try:
    import cocoex
except ImportError:
    print('BBOB COCO not found. COCO benchmarks will not be available')

import scipy.stats
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def run_ea(ea):
    result = ea.one_run()
    return result

def run_hill_climber(fitness_function, iterations):
    result = fitness_function.hill_climber(iterations)
    return result

class EAResult:
    # a class for holding the results of a single run of an EA
    def __init__(self):
        self.eval_counts = []
        self.average_fitness = dict()
        self.best_fitnesses = dict()
        self.final_average_fitness = None
        self.final_best_fitness = None
        self.termination_reason = None

        self.selection_function_name = None
        self.selection_function_id = None
        
        self.fitness_function_name = None
        self.fitness_function_id = None

class EAResultCollection:
    # a class for holding the results of several EA runs
    def __init__(self, results=None):
        self.results = []
        self.fitness_functions = set()
        self.selection_functions = set()

        if results is not None:
            self.add(results)

    def add(self, new_results):
        # adds a list of new results to the result collection
        self.results.extend(new_results)
        for r in new_results:
            self.fitness_functions.add(r.fitness_function)
            self.selection_functions.add(r.selection_function)

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
        new_collection = EAResultCollection(filtered_results)
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
        self.trap_size = None

        self.fitness_function_a = None
        self.epistasis_k = None

        self.fitness_function_offset = None
        self.loci_values = None
        self.epistasis = None

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

        self.name = config.get('EA', 'fitness function')
        self.genome_type = self.genome_types.get(self.name, None)

        self.genome_length = config.getint('fitness function', 'genome length')
        self.fitness_function_a = config.getfloat('fitness function', 'a')
        self.max_initial_range = config.getfloat('fitness function', 'max initial range')
        self.trap_size = config.getint('fitness function', 'trap size')
        self.epistasis_k = config.getint('fitness function', 'epistasis k')
        self.epistasis_m = config.getint('fitness function', 'epistasis m')

        self.coco_function_index = config.getint('fitness function', 'coco function index')

        if self.name == 'rastrigin':
            self.fitness_function_offset = self.generate_offset(self.genome_length)
        else:
            self.fitness_function_offset = None

        if self.name == 'nk_landscape':
            self.loci_values, self.epistasis = self.generate_epistatis(self.genome_length, self.epistasis_k)
        elif self.name == 'mk_landscape':
            self.loci_values, self.epistasis = self.generate_mk_epistatis(self.genome_length, self.epistasis_m, self.epistasis_k)
        else:
            self.loci_values, self.epistasis = None, None

        self.assign_id()

    def start(self):
        # should be called once at the start of each search
        if self.name == 'coco':
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
            if self.genome_type == 'bool':
                genome.append(random.choice((True, False)))
            elif self.genome_type == 'float':
                genome.append(random.uniform(-self.max_initial_range, self.max_initial_range))
            else:
                print('WARNING: genome type {0} not recognized for random genome generation'.format(self.genome_type))
                break

        # return the new genome
        return genome

    def hill_climber(self, max_evals):
        # performs a first-ascent hill climb, restarting with random genomes at local optima
        # for real-valued genomes, hill climbing is done with stochastic perturbations and may not settle on the local optima
        # returns an EAResults object with results, and the best genome found

        self.start()

        # generate and evaluate a new genome
        current_genome = self.random_genome()
        current_fitness = self.evaluate(current_genome)

        # evaluate adjacent genomes, keeping improvements and restarting on stagnation
        evals = 1
        best_fitness = current_fitness
        best_genome = current_genome

        best_fitnesses = dict()
        best_fitnesses[1] = best_fitness

        # bool and real genomes are treated slightly differently
        if self.genome_type == 'bool':
            while evals < max_evals:
                climbing = False
                for i in random.sample(range(self.genome_length), self.genome_length):
                    # copy the genome and flip one bit
                    new_genome = current_genome[:]
                    new_genome[i] = not new_genome[i]

                    # rate the genome. If it is better, replace the current, and start flipping bits in a new order
                    new_fitness = self.evaluate(new_genome)
                    evals += 1

                    if new_fitness > current_fitness:
                        current_genome = new_genome
                        current_fitness = new_fitness
                        best_fitnesses[evals] = current_fitness
                        climbing = True
                        break

                # if we go through all the genes and have not climbed at all, we have found a local optimum. generate a
                # new genome and start another hill climb
                if not climbing:
                    best_fitness = current_fitness
                    best_genome = current_genome
                    current_genome = self.random_genome()

            # after we have expended all evaluations, record the best fitness and genome
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_fitnesses[evals] = best_fitness
                best_genome = current_genome

        elif self.genome_type == 'float':
            while evals < max_evals:
                climbing = False
                # the max perturbation is equal to the greatest absolute value in the genome
                max_perturbation = max(abs(c) for c in current_genome)

                for i in random.sample(range(self.genome_length), self.genome_length):

                    # copy the genome
                    new_genome = current_genome[:]

                    # generate a positive perturbation
                    perturbation = random.triangular(0.0, max_perturbation, 0.0)

                    # apply the perturbation
                    new_genome[i] += perturbation

                    # rate the genome. If it is better, replace the current, and start perturbing bits in a new order
                    new_fitness = self.evaluate(new_genome)
                    evals +=  1

                    if new_fitness > current_fitness:
                        current_genome = new_genome
                        current_fitness = new_fitness
                        best_fitnesses[evals] = current_fitness
                        climbing = True
                        break

                    # if a positive perturbation didn't increase fitness, try a negative one
                    else:
                        perturbation = random.triangular(-1 * max_perturbation, 0.0, 0.0)

                        # apply the perturbation
                        new_genome[i] += perturbation

                        # rate the genome. If it is better, replace the current, and start perturbing bits in a new order
                        new_fitness = self.evaluate(new_genome)
                        evals += 1

                        if new_fitness > current_fitness:
                            current_genome = new_genome
                            current_fitness = new_fitness
                            climbing = True
                            break

                # if we go through all the genes and have not climbed at all, we have found a local optimum. generate a
                # new genome and start another hill climb
                if not climbing:
                    best_fitness = current_fitness
                    best_genome = current_genome
                    current_genome = self.random_genome()

            # after we have expended all evaluations, record the best fitness and genome
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_fitnesses[evals] = best_fitness
                best_genome = current_genome

        else:
            print('WARNING: genome type {0} not recognized for hill climber'.format(self.genome_type))

        self.finish()

        results = EAResult()
        results.eval_counts = list(best_fitnesses.keys())
        results.best_fitnesses = best_fitnesses
        results.final_best_fitness = best_fitness

        # create a dummy selection function for the hill climber results
        selection_function = SelectionFunction()
        selection_function.type = 'hill_climber'
        selection_function.display_name = 'hill_climber'
        selection_function.id = 'hill_climber'
        results.selection_function = selection_function

        results.fitness_function = self

        # return best fitness and genome
        return results

    def fitness_target_hit(self):
        if self.name == 'coco':
            return self.coco_function.final_target_hit
        else:
            return False

    def rosenbrock(self, x, a):
        result = 0
        n = len(x)
        for i in range(n-1):
            if -5 <= x[i] <= 10:
                result += (1 - x[i])**2 + a*(x[i+1] - x[i]**2)**2

        result = -1 * result
        return result

    def rastrigin(self, x, a):
        n = len(x)
        result = a * n
        for i in range(n):
            result += x[i]**2 - a*math.cos(2*math.pi*x[i])

        result = -1 * result
        return result

    def dtrap(self, x, trap_size):
        result = 0
        for i in range(0, self.genome_length, trap_size):
            trap = x[i:i+trap_size]
            if all(trap):
                result += trap_size
            else:
                result += trap.count(False)

        return result

    def nk_landscape(self, x):
        result = 0

        for i in range(self.genome_length):
            locus = [x[i]]
            locus.extend((x[j] for j in self.epistasis[i]))
            locus_fitness = self.loci_values[tuple(locus)]
            result += locus_fitness

        return result

    def mk_landscape(self, x):
        result = 0

        for e in self.epistasis:
            locus = []
            locus.extend(x[j] for j in e)
            locus_fitness = self.loci_values[tuple(locus)]
            result += locus_fitness

        return result

    def coco(self, x):
        # multiply by -1 since coco functions are minimization functions
        return -1 * self.coco_function(x)

    def offset_rastrigin(self, x, a, offset):
        offset_x = list(x)
        for i in range(len(x)):
            offset_x[i] += offset[i]
        return self.rastrigin(offset_x, a)

    def generate_offset(self, n):
        result = []
        for _ in range(n):
            result.append(random.choice([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
        return result

    def generate_epistatis(self, n, k):
        loci_values = dict()
        for locus in itertools.product([True, False], repeat=k+1):
            loci_values[locus] = random.randint(0,k)

        epistasis = dict()
        for i in range(n):
            epistasis[i] = sorted(random.sample(list(j for j in range(n) if j != i), k))

        return loci_values, epistasis

    def generate_mk_epistatis(self, n, m, k):
        loci_values = dict()
        for locus in itertools.product([True, False], repeat=k):
            loci_values[locus] = random.randint(0,k)

        epistasis = list()
        for _ in range(m):
            epistasis.append(list(random.sample(list(j for j in range(n)), k)))

        return loci_values, epistasis

    def evaluate(self, genome):
        if self.name == 'rosenbrock':
            fitness = self.rosenbrock(genome, self.fitness_function_a)
        elif self.name == 'rastrigin':
            fitness = self.offset_rastrigin(genome, self.fitness_function_a, self.fitness_function_offset)
        elif self.name == 'dtrap':
            fitness = self.dtrap(genome, self.trap_size)
        elif self.name == 'nk_landscape':
            fitness = self.nk_landscape(genome)
        elif self.name == 'mk_landscape':
            fitness = self.mk_landscape(genome)
        elif self.name == 'coco':
            fitness = self.coco(genome)

        else:
            raise Exception('EPPSEA BasicEA ERROR: fitness function name {0} not recognized'.format(self.name))

        return fitness


class SelectionFunction:
    def __init__(self):
        self.eppsea_selection_function = None
        
        self.parent_selection_type = None
        self.parent_selection_tournament_k = None
        
        self.survival_selection_type = None
        self.survival_selection_tournament_k = None
        
        self.display_name = None

    def __hash__(self):
        return hash(self.id)
        
    def assign_id(self):
        # assigns a random id to self. Every unique Selection Function should call this once
        self.id = '{0}_{1}_{2}'.format('SelectionFunction', str(id(self)), str(uuid.uuid4()))

    def generate_from_config(self, config):
        if config.getboolean('selection function', 'evolved'):
            file_path = config.get('survival selection function', 'file path (for evolved selection)')
            self.eppsea_selection_function = eppsea_base.GPTree()
            self.eppsea_selection_function.load_from_dict(file_path)
            
        else:
            self.parent_selection_type = config.get('selection function', 'parent selection type')
            if self.parent_selection_type == 'k_tournament':
                self.parent_selection_tournament_k = config.getint('selection function', 'parent selection tournament k')
    
            self.survival_selection_type = config.get('selection function', 'survival selection type')
            if self.survival_selection_type == 'k_tournament':
                self.survival_selection_tournament_k = config.getint('selection function', 'survival selection tournament k')
            
        self.display_name = config.get('selection function', 'display name')
        
        self.assign_id()

    def generate_from_eppsea_individual(self, eppsea_selection_function):
        self.type = 'eppsea_selection_function'
        self.display_name = 'Evolved Selection Function'
        self.tournament_k = None
        self.eppsea_selection_function = eppsea_selection_function
        self.assign_id()
        
    def parent_selection(self, population, n, generation_number):
        if self.eppsea_selection_function is not None:
            return self.eppsea_selection_function.select_parents(population, n, generation_number)
        else:
            return self.basic_selection(population, n, self.parent_selection_type, self.parent_selection_tournament_k)
        
    def survival_selection(self, population, n, generation_number):
        if self.eppsea_selection_function is not None:
            return self.eppsea_selection_function.select_survivors(population, n, generation_number)
        else:
            return self.basic_selection(population, n, self.survival_selection_type, self.survival_selection_tournament_k)

    def basic_selection(self, population, n, type, tournament_k):
        if type == 'truncation':
            return sorted(population, key=lambda x: x.fitness)[:n]

        elif type == 'fitness_rank':
            selected = []
            sorted_population = sorted(population, key=lambda p: p.fitness, reverse=True)
            ranks = list(range(len(population), 0, -1))
            sum_ranks = (len(sorted_population) * (len(sorted_population)+1)) / 2
            for _ in range(n):
                r = random.randint(0, sum_ranks)
                i = 0
                while r > ranks[i]:
                    r -= ranks[i]
                    i += 1
                selected.append(population[i])
            return selected

        elif type == 'fitness_proportional':
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

        elif type == 'k_tournament':
            selected = []
            for _ in range(n):
                tournament = random.sample(population, tournament_k)
                winner = max(tournament, key=lambda p: p.fitness)
                selected.append(winner)
            return selected

        elif type == 'random':
            selected = []
            for _ in range(n):
                selected.append(random.choice(population))
            return selected

        elif type == 'stochastic_universal_sampling':
            # normalize the weights, if necessary
            normalized_weights = list(p.fitness for p in population)
            min_weight = min(normalized_weights)
            if min_weight < 0:
                for i in range(len(normalized_weights)):
                    normalized_weights[i] -= min_weight

            # build a list of the indices and cumulative selection weights
            indices_and_weights = []
            cum_weight = 0
            for i in range(len(population)):
                cum_weight += normalized_weights[i]
                indices_and_weights.append((i, cum_weight))
            sum_weight = cum_weight

            # if the sum weight is 0 or inf, just return random candidate
            if sum_weight == 0 or sum_weight == math.inf:
                selected = []
                for _ in range(n):
                    selected.append(random.choice(population))
                return selected

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
            print('BASIC SELECTION {0} NOT FOUND'.format(type))


class EA:
    def __init__(self, config, fitness_function, selection_function):
        self.mu = config.getint('EA', 'population size')
        self.lam = config.getint('EA', 'offspring size')

        self.mutation_rate = config.getfloat('EA', 'mutation rate')
        self.max_evals = config.getint('EA', 'maximum evaluations')
        self.terminate_on_fitness_convergence = config.getboolean('EA', 'terminate on fitness convergence')
        self.generations_to_convergence = config.getint('EA', 'generations to convergence')
        self.terminate_at_target_fitness = config.getboolean('EA', 'terminate at target fitness')
        self.use_custom_target_fitness = config.getboolean('EA', 'use custom target fitness')
        self.target_fitness = config.getfloat('EA', 'target fitness')
        self.terminate_on_population_convergence = config.getboolean('EA', 'terminate on population convergence')
        self.population_convergence_threshold = config.getfloat('EA', 'population convergence threshold')

        self.fitness_function = fitness_function
        self.selection_function = selection_function

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
            all_parents = self.selection_function.parent_selection(population, num_parents, generation_number)

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
            survivors = self.selection_function.survival_selection(population, self.mu, generation_number)
            population = survivors

            # record average and best fitness
            average_fitness = statistics.mean(p.fitness for p in population)
            average_fitnesses[evals] = average_fitness

            best_fitness = max(p.fitness for p in population)
            best_fitnesses[evals] = best_fitness

            generation_number += 1

            # check for termination conditions
            if self.terminate_at_target_fitness:
                if self.use_custom_target_fitness:
                    if any(p.fitness >= self.target_fitness for p in population):
                        termination_reason = 'hit_target_fitness'
                        break
                else:
                    if self.fitness_function.fitness_target_hit():
                        termination_reason = 'hit_target_fitness'
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

        run_results.selection_function = self.selection_function
        run_results.selection_function_id = self.selection_function.id

        run_results.fitness_function = self.fitness_function
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
        self.test_hill_climber = config.getboolean('EA', 'test hill climber')
        self.hill_climber_iterations = config.getint('EA', 'hill climber iterations')
        self.fitness_function_name = config.get('EA', 'fitness function')
        self.use_multiobjective_ea = config.getboolean('EA', 'use multiobjective ea')

        self.num_training_fitness_functions = None
        self.num_testing_fitness_functions = None
        self.training_fitness_functions = None
        self.testing_fitness_functions = None

        self.eppsea = None

        self.prepare_fitness_functions(config)

        self.basic_selection_functions = None
        self.prepare_basic_selection_functions(config)

        self.basic_results = None

        self.eppsea_fitness_assignment_method = 'best_fitness_reached'

    def prepare_fitness_functions(self, config):
        # generates the fitness functions to be used in the EAs

        # if we are using coco, count the number of available function indices
        if self.fitness_function_name == 'coco':
            genome_length = config.getint('fitness function', 'genome length')
            if genome_length not in [2,3,5,10,20,40]:
                print('WARNING: genome length {0} may not be supported by coco'.format(genome_length))
            coco_function_index = config.get('fitness function', 'coco function index')
            suite = cocoex.Suite('bbob', '', 'dimensions:{0}, function_indices:{1}'.format(genome_length, coco_function_index))
            coco_ids = list(suite.ids())
        else:
            coco_ids = None

        # get the number of training and testing fitness functions to be used
        self.num_training_fitness_functions = config.getint('EA', 'num training fitness functions')
        if config.getboolean('EA', 'test generalization'):
            self.num_testing_fitness_functions = config.getint('EA', 'num testing fitness functions')
            # if we are using coco and testing fitness functions is -1, automatically use remaining instances as test functions
            if self.num_testing_fitness_functions == -1 and self.fitness_function_name == 'coco':
                self.num_testing_fitness_functions = len(coco_ids) - self.num_training_fitness_functions
        else:
            self.num_testing_fitness_functions = 0

        self.training_fitness_functions = []
        training_fitness_function_path = config.get('EA', 'fitness function training instances directory')
        self.testing_fitness_functions = []
        testing_fitness_function_path = config.get('EA', 'fitness function testing instances directory')

        # shuffle coco indeces so there is no bias in assigning training vs testing functions
        if coco_ids is not None:
            random.shuffle(coco_ids)

        if config.getboolean('EA', 'generate new fitness functions'):
            for i in range(self.num_training_fitness_functions):
                new_fitness_function = FitnessFunction()
                new_fitness_function.generate(config)
                if self.fitness_function_name == 'coco':
                    new_fitness_function.coco_function_id = coco_ids.pop()
                self.training_fitness_functions.append(new_fitness_function)

            for i in range(self.num_testing_fitness_functions):
                new_fitness_function = FitnessFunction()
                new_fitness_function.generate(config)
                if self.fitness_function_name == 'coco':
                    new_fitness_function.coco_function_id = coco_ids.pop()
                self.testing_fitness_functions.append(new_fitness_function)

            if config.getboolean('EA', 'save generated fitness functions'):
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
                raise Exception('ERROR: Attempting to load fitness functions from non-existent path {0}'.format(training_fitness_function_path))

            training_fitness_function_files = sorted(os.listdir(training_fitness_function_path))
            for filepath in training_fitness_function_files:
                try:
                    full_filepath = '{0}/{1}'.format(training_fitness_function_path, filepath) 
                    with open(full_filepath, 'rb') as file:
                        self.training_fitness_functions.append(pickle.load(file))
                except (pickle.PickleError, pickle.PickleError, ImportError, AttributeError):
                    print('Failed to load fitness function at {0}, possibly not a saved fitness function'.format(filepath))
                    pass
                
                if len(self.training_fitness_functions) == self.num_training_fitness_functions:
                    break
                    
            if config.getboolean('EA', 'test generalization'):
                
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

    def run_eas(self, eas, is_testing):
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

        return EAResultCollection(all_run_results)

    def run_hill_climbers(self, fitness_functions, iterations, is_testing):
        # runs hill climbers on a set of fitness functions, once for every run
        # runs each of the eas for the configured number of runs, returning one result_holder for each ea
        all_hill_climber_results = []

        if is_testing:
            runs = self.testing_runs
        else:
            runs = self.training_runs

        if self.using_multiprocessing:
            # setup parameters for multiprocessing
            params = []
            for f in fitness_functions:
                params.extend([(f, iterations)]*runs)
            # run all runs
            pool = multiprocessing.Pool()
            results = pool.starmap(run_hill_climber, params)
            pool.close()

            # split up results by ea
            for i, f in enumerate(fitness_functions):
                start = i * runs
                stop = (i + 1) * runs

                hill_climber_results = list(r for r in results[start:stop])

                all_hill_climber_results.extend(hill_climber_results)

        else:
            for f in fitness_functions:
                for r in range(runs):
                    hill_climber_results = run_hill_climber(f, iterations)
                    all_hill_climber_results.append(hill_climber_results)

        return EAResultCollection(all_hill_climber_results)

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

        eas = self.get_eas(fitness_functions, selection_functions)
        ea_results = self.run_eas(eas, False)
        self.assign_eppsea_fitness(selection_functions, ea_results)

    def assign_eppsea_fitness(self, selection_functions, ea_results):
        # takes an EAResultsCollection and uses it to assign fitness values to the eppsea_population

        # when Eppsea_basicEA starts, eppsea fitness is assigned by average best fitness reached on the bottom-level EA
        # if we are currently assigning eppsea fitness by looking at average best fitness reached, and at least 10% of
        # the runs are reaching the fitness goal, switch to assigning eppsea fitness by proportion of time
        # the fitness goal is found
        # if we are currently assigning by that, and at least 95% of the runs are reaching the fitness goal, switch
        # to assigning by number of evals needed to reach the fitness goal
        # in the case of either switch, all population members' fitnesses become -infinity before fitness assignment
        # this effectively sets the fitness of all old eppsea population members to -infinity
        # the counters for average/best fitness change at the eppsea level are also manually reset, since this counts as a fitness improvement
        if self.eppsea_fitness_assignment_method == 'best_fitness_reached':
            if len(list(r for r in ea_results.results if r.termination_reason == 'target_fitness_hit')) / len (ea_results.results) >= 0.1:
                self.log('At eval count {0}, eppsea fitness assignment changed to proportion_hitting_target_fitness'.format(self.eppsea.gp_evals))
                self.eppsea_fitness_assignment_method = 'proportion_hitting_target_fitness'
                for p in self.eppsea.population:
                    p.fitness = -math.inf
                self.eppsea.gens_since_avg_fitness_improvement = 0
                self.eppsea.gens_since_best_fitness_improvement = 0
                self.eppsea.highest_average_fitness = -math.inf
                self.eppsea.highest_best_fitness = -math.inf
        if self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
            if len(list(r for r in ea_results.results if r.termination_reason == 'target_fitness_hit')) / len(ea_results.results) == 1.0:
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
                # assign fitness as the average of the average final best fitnesses or, if multiobjective ea is on, the list of average final best fitnesses
                if self.use_multiobjective_ea:
                    s.eppsea_selection_function.mo_fitnesses = average_final_best_fitnesses
                else:
                    s.eppsea_selection_function.fitness = statistics.mean(average_final_best_fitnesses)

            elif self.eppsea_fitness_assignment_method == 'proportion_hitting_target_fitness':
                # assign the fitness as the proportion of runs that hit the target fitness
                s.eppsea_selection_function.fitness = len(list(r for r in s_results.results if r.termination_reason == 'target_fitness_hit')) / len (s_results.results)

            elif self.eppsea_fitness_assignment_method == 'evals_to_target_fitness':
                # loop through all fitness functions to get average evals to target fitness
                all_final_evals = []
                for fitness_function in s_results.fitness_functions:
                    fitness_function_results = s_results.filter(fitness_function=fitness_function)
                    final_evals = (max(r.evals) for r in fitness_function_results)
                    all_final_evals.append(statistics.mean(final_evals))
                # assign fitness as -1 * the average of final eval counts
                s.eppsea_selection_function.fitness = statistics.mean(all_final_evals)
            else:
                raise Exception('ERROR: fitness assignment method {0} not recognized by eppsea_basicEA'.format(self.eppsea_fitness_assignment_method))

    def test_against_basic_selection(self, eppsea_individuals):

        selection_functions = self.basic_selection_functions
        if self.config.getint('EA', 'offspring size') > self.config.getint('EA', 'population size') * 2:
            for s in list(selection_functions):
                if s.type == 'truncation':
                    selection_functions.remove(s)

        eppsea_selection_functions = []

        for eppsea_individual in eppsea_individuals:
            eppsea_selection_function = SelectionFunction()
            eppsea_selection_function.generate_from_eppsea_individual(eppsea_individual)
            selection_functions.append(eppsea_selection_function)

        if self.test_generalization:
            fitness_functions = self.testing_fitness_functions
        else:
            fitness_functions = self.training_fitness_functions

        eas = self.get_eas(fitness_functions, selection_functions)

        ea_results = self.run_eas(eas, True)

        if self.test_hill_climber:
            hill_climber_results = self.run_hill_climbers(self.testing_fitness_functions, self.hill_climber_iterations, True)
            full_results = EAResultCollection(ea_results.results + hill_climber_results.results)
        else:
            full_results = ea_results

        return full_results

    def export_run_results(self, final_test_results):
        # takes a list of ResultsHolder objects and exports several json files that are compatible with the post_process
        # script
        results_paths = []
        # get a set of the selection functions tested
        selection_function_names = set(f.selection_function_name for f in final_test_results.results)
        # create a dictionary of results for each selection function
        for s in selection_function_names:
            results_dict = dict()
            results_dict['Name'] = s
            # get the list of results for this particular selection function
            results = list(f for f in final_test_results.results if f.selection_function_name == s)
            # check if a log scale needs to be used
            if any(r.fitness_function_name in ['rosenbrock', 'rastrigin', 'coco'] for r in results):
                results_dict['Log Scale'] = True
            else:
                results_dict['Log Scale'] = False
            # check if this is the EPPSEA selection function, in which case a t test must be done
            if s == 'eppsea_selection_function':
                results_dict['T Test'] = True
            else:
                results_dict['T Test'] = False
            results_dict['Fitness Functions'] = dict()
            # enumerate over all fitness functions tested
            for i,f in enumerate(self.testing_fitness_functions):
                # get the result holder corresponding to the runs for this fitness function
                fitness_function_result = next(r for r in results if r.fitness_function_id == f.id)
                # get the mappings of evaluations to best fitnesses for all runs of this fitness function
                run_results = list(r['best_fitnesses'] for r in fitness_function_result)

                fitness_function_name = f.fitness_function_name
                results_dict['Fitness Functions'][i] = {'Name':fitness_function_name, 'Runs':run_results}

            # export the json file and record its location
            results_path = '{0}/{1}.json'.format(self.results_directory, s)
            with open(results_path, 'w') as file:
                json.dump(results_dict, file)
            results_paths.append(results_path)
        return results_paths

    def postprocess(self, results):
        # runs postprocessing on an EAResultCollection results
        output = ''
        # Analyze results for each fitness function
        for fitness_function in results.fitness_functions:
            plt.clf()
            output += 'Analyzing results for fitness function with id {0} ---------------------------------\n'.format(
                fitness_function.id)
            output += 'Plotting figure\n'
            # Get the name of the fitness function from one of the result files
            fitness_function_name = fitness_function.name

            # filter out the results for this fitness function
            fitness_function_results = results.filter(fitness_function=fitness_function)

            # Set the plot to use Log Scale if any of the result files require it
            if any(s in fitness_function_name for s in ['rastrigin', 'rosenbrock', 'coco']):
                plt.yscale('symlog')

            # Plot results for each selection function
            for selection_function in fitness_function_results.selection_functions:
                selection_function_results = fitness_function_results.filter(selection_function=selection_function)
                selection_function_name = selection_function.display_name
                mu = selection_function_results.get_eval_counts()
                average_best_fitnesses = []
                for m in mu:
                    average_best_fitnesses.append(statistics.mean(r.best_fitnesses[m] for r in selection_function_results.results if m in r.best_fitnesses))

                plt.plot(mu, average_best_fitnesses, label=selection_function_name)

            plt.xlabel('Evaluations')
            plt.ylabel('Best Fitness')
            plt.legend(loc=(1.02, 0))
            plt.savefig('{0}/figure_{1}.png'.format(self.results_directory, fitness_function.id),
                        bbox_inches='tight')

            output += 'Plotting boxplot\n'
            final_best_fitnesses_list = []
            selection_name_list = []

            # Set the plot to use Log Scale if any of the result files require it
            if any(s in fitness_function_name for s in ['rastrigin', 'rosenbrock', 'coco']):
                plt.yscale('symlog')

            for selection_function in fitness_function_results.selection_functions:
                selection_function_results = fitness_function_results.filter(selection_function=selection_function)
                selection_function_name = selection_function.display_name
                selection_name_list.append(selection_function_name)
                final_best_fitnesses = list(r.final_best_fitness for r in selection_function_results.results)
                final_best_fitnesses_list.append(final_best_fitnesses)
            plt.boxplot(final_best_fitnesses_list, labels=selection_name_list)

            plt.xlabel('Evaluations')
            plt.xticks(rotation=90)
            plt.ylabel('Final Best Fitness')
            legend = plt.legend([])
            legend.remove()
            plt.savefig('{0}/boxplot_{1}.png'.format(self.results_directory, fitness_function.id),
                        bbox_inches='tight')

            output += 'Doing t-tests\n'
            for selection_function1 in fitness_function_results.selection_functions:
                selection_function_results1 = fitness_function_results.filter(selection_function=selection_function1)
                selection_function_name1 = selection_function1.display_name
                final_best_fitnesses1 = list(r.final_best_fitness for r in selection_function_results1.results)
                final_evals1 = list(max(r.eval_counts) for r in selection_function_results1.results)
                # round means to 5 decimal places for cleaner display
                average_final_best_fitness1 = round(statistics.mean(final_best_fitnesses1), 5)
                average_final_evals1 = round(statistics.mean(final_evals1), 5)
                output += 'Mean performance of {0}: {1}, in {2} evals\n'.format(selection_function_name1, average_final_best_fitness1, final_evals1)
                # perform a t test with all the other results if this is an evolved selection function
                if selection_function1.eppsea_selection_function is not None:
                    for selection_function2 in fitness_function_results.selection_functions:
                        if selection_function2 is not selection_function1:
                            selection_function_results2 = fitness_function_results.filter(selection_function=selection_function2)
                            selection_function_name2 = selection_function2.display_name
                            final_best_fitnesses2 = list(r.final_best_fitness for r in selection_function_results2.results)
                            final_evals2 = list(max(r.eval_counts) for r in selection_function_results2.results)
                            average_final_best_fitness2 = round(statistics.mean(final_best_fitnesses2), 5)
                            average_final_evals2 = round(statistics.mean(final_evals2), 5)
                            _, p_fitness = scipy.stats.ttest_rel(final_best_fitnesses1, final_best_fitnesses2)
                            _, p_evals = scipy.stats.ttest_rel(final_evals1, final_evals2)
                            mean_difference_fitness = round(average_final_best_fitness1 - average_final_best_fitness2, 5)
                            mean_difference_evals = round(average_final_evals1 - average_final_evals2, 5)

                            output += '\tMean performance of {0}: {1}, in {2} evals\n'.format(selection_function_name2, average_final_best_fitness2, average_final_evals2)

                            if p_fitness < 0.05:
                                if mean_difference_fitness > 0:
                                    output += '\t\t{0} performed {1} better | p-value: {2}\n'.format(selection_function_name1, mean_difference_fitness, p_fitness)
                                else:
                                    output += '\t\t{0} performed {1} worse | p-value: {2}\n'.format(selection_function_name1, mean_difference_fitness, p_fitness)
                            else:
                                output += '\t\t{0} performance difference is insignificant | p-value: {1}\n'.format(selection_function_name1, p_fitness)

                            if p_evals < 0.05:
                                if mean_difference_evals < 0:
                                    output += '\t\t{0} used {1} fewer evals | p-value: {2}\n'.format(selection_function_name1, mean_difference_evals, p_evals)
                                else:
                                    output += '\t\t{0} used {1} more evals | p-value: {2}\n'.format(selection_function_name1, mean_difference_evals, p_evals)
                            else:
                                output += '\t\t{0} eval count difference is insignificant | p-value: {1}\n'.format(selection_function_name1, p_evals)

        return output


    def run_eppsea_basicea(self):
        print('Now starting EPPSEA')
        start_time = time.time()

        eppsea_config = self.config.get('EA', 'base eppsea config path')
        eppsea = eppsea_base.Eppsea(eppsea_config)
        self.eppsea = eppsea

        eppsea.start_evolution()

        while not eppsea.evolution_finished:
            self.evaluate_eppsea_population(eppsea.new_population, False)
            eppsea.next_generation()

        if self.use_multiobjective_ea:
            best_selection_functions = eppsea.final_best_members
        else:
            best_selection_functions = [eppsea.final_best_member]
        print('Running final tests')
        final_test_results = self.test_against_basic_selection(best_selection_functions)
        end_time = time.time() - start_time
        self.log('Total time elapsed: {0}'.format(end_time))

        print('Running Postprocessing')
        postprocess_results = self.postprocess(final_test_results)
        self.log('Postprocess results:')
        self.log(postprocess_results)

        eppsea_base_results_path = eppsea.results_directory
        shutil.copytree(eppsea_base_results_path, self.results_directory + '/base')


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = EppseaBasicEA(config)
    shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))
    evaluator.run_eppsea_basicea()

    pickle_path = '{0}/EppseaBasicEA'.format(evaluator.results_directory)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(evaluator, pickle_file)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    main(config_path)
