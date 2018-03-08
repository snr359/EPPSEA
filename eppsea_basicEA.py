import math
import random
import statistics
import itertools
import sys
import configparser
import os
import shutil
import csv
import subprocess
import multiprocessing
import time
import uuid
import datetime
import pickle

import eppsea_base

def postprocess(final_results_path, results_directory):
    # calls the postprocessing script on a pickled dictionary mapping fitness functions to ResultHolder objects
    params = ['python3', 'post_process.py', final_results_path, results_directory]
    result = subprocess.run(params, stdout=subprocess.PIPE, universal_newlines=True)

    return result.stdout

def t_test(a, b):
    # does a t-test between data sets a and b. effectively just calls another script, but does so in a separate
    # process instead of importing that script, to keep this one compatible with pypy
    filename = 'temp_{0}.csv'.format(str(uuid.uuid4()))
    with open(filename, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(a)
        writer.writerow(b)
    t_test_results = subprocess.check_output(['python3', 't_test.py', filename])
    _, _, t, p_value = list(float(r) for r in t_test_results.split())
    os.remove(filename)
    return t, p_value

def evaluate_eppsea_population(basic_ea, eppsea_population, using_multiprocessing):
    # evaluates a population of eppsea individuals and assigns fitness values to them

    if using_multiprocessing:
        # setup parameters for multiprocessing
        params = []
        for p in eppsea_population:
            params.extend([('eppsea_selection_function', p)]*basic_ea.runs)

        # run all runs
        pool = multiprocessing.Pool()
        results_all_runs = pool.starmap(basic_ea.one_run, params)
        pool.close()

    else:
        results_all_runs = []
        for p in eppsea_population:
            for r in range(basic_ea.runs):
                results_all_runs.append(basic_ea.one_run('eppsea_selection_function', p))

    for i, p in enumerate(eppsea_population):
        start = i*basic_ea.runs
        stop = (i+1)*basic_ea.runs
        run_results = results_all_runs[start:stop]
        p.fitness = statistics.mean(r['final_best_fitness'] for r in run_results)

def test_against_basic_selection(basic_ea, eppsea_selection_function):
    parent_selection_functions = ['truncation', 'fitness_proportional', 'fitness_rank', 'k_tournament']
    results_all_selections = dict()
    if basic_ea.lam > basic_ea.mu * 2:
        parent_selection_functions.remove('truncation')

    # set up parameters for multiprocessing
    params = []
    for parent_selection_function in parent_selection_functions:
        params.extend([(parent_selection_function, None)]*basic_ea.runs)

    params.extend([('eppsea_selection_function', eppsea_selection_function)]*basic_ea.runs)

    pool = multiprocessing.Pool()
    results_all_runs = pool.starmap(basic_ea.one_run, params)
    pool.close()

    for i, parent_selection_function in enumerate(parent_selection_functions):
        start = i*basic_ea.runs
        stop = (i+1)*basic_ea.runs
        new_result_holder = ResultHolder()
        new_result_holder.selection_function = parent_selection_function
        new_result_holder.fitness_function = basic_ea.fitness_function
        new_result_holder.run_results = results_all_runs[start:stop]
        results_all_selections[parent_selection_function] = new_result_holder

    new_result_holder = ResultHolder()
    new_result_holder.selection_function = 'eppsea_selection_function'
    new_result_holder.fitness_function = basic_ea.fitness_function
    new_result_holder.run_results = results_all_runs[-basic_ea.runs:]
    results_all_selections['eppsea_selection_function'] = new_result_holder

    return results_all_selections

class ResultHolder:
    # a class for holding the results of eppsea runs
    def __init__(self):
        self.selection_function = None
        self.fitness_function = None
        self.run_results = []

    def get_eval_counts(self):
        return sorted(self.run_results[0]['best_fitnesses'].keys())

    def get_average_average_fitness(self):
        average_average_fitnesses = []
        for m in self.get_eval_counts():
            average_average_fitnesses.append(statistics.mean(r['average_fitnesses'][m] for r in self.run_results))
        return average_average_fitnesses

    def get_average_best_fitness(self):
        average_best_fitnesses = []
        for m in self.get_eval_counts():
            average_best_fitnesses.append(statistics.mean(r['best_fitnesses'][m] for r in self.run_results))

        return average_best_fitnesses

    def get_final_average_fitness_all_runs(self):
        average_fitnesses_all_runs = list(r['final_average_fitness'] for r in self.run_results)
        return average_fitnesses_all_runs

    def get_final_best_fitness_all_runs(self):
        best_fitnesses_all_runs = list(r['final_best_fitness'] for r in self.run_results)
        return best_fitnesses_all_runs

    def get_average_final_best_fitness(self):
        return statistics.mean(self.get_final_best_fitness_all_runs())

    def get_average_final_average_fitness(self):
        return statistics.mean(self.get_final_average_fitness_all_runs())


class basicEA:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool'
    }

    def __init__(self, config):
        self.mu = config.getint('EA', 'population size')
        self.lam = config.getint('EA', 'offspring size')

        self.mutation_rate = config.getfloat('EA', 'mutation rate')
        self.max_evals = config.getint('EA', 'maximum evaluations')

        self.runs = config.getint('EA', 'runs')

        self.fitness_function = config.get('EA', 'fitness function')
        self.genome_type = self.genome_types.get(self.fitness_function, None)

        self.genome_length = config.getint('fitness function', 'genome length')
        self.fitness_function_a = config.getfloat('fitness function', 'a')
        self.max_initial_range = config.getfloat('fitness function', 'max initial range')
        self.trap_size = config.getint('fitness function', 'trap size')
        self.epistasis_k = config.getint('fitness function', 'epistasis k')
        self.tournament_k = config.getint('EA', 'tournament k')

        if self.fitness_function == 'rastrigin':
            self.fitness_function_offset = self.generate_offset(self.genome_length)
        else:
            self.fitness_function_offset = None

        if self.fitness_function == 'nk_landscape':
            self.loci_values, self.epistasis = self.generate_epistatis(self.genome_length, self.epistasis_k)
        else:
            self.loci_values, self.epistasis = None, None

        self.basic_results = None

        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_basicEA_" + str(present_time)

        self.results_directory = 'results/eppsea_basicEA/{0}'.format(experiment_name)
        os.makedirs(self.results_directory, exist_ok=True)

        self.log_file_location = '{0}/log.txt'.format(self.results_directory)

    def log(self, message):
        print(message)
        with open(self.log_file_location, 'a') as log_file:
            log_file.write(message + '\n')

    def plot_results(self, results):
        filename = 'temp{0}'.format(str(uuid.uuid4()))
        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if self.fitness_function in [
                'rosenbrock',
                'rastrigin'
                ]:
                symlog = '1'
            else:
                symlog = '0'

            selection_functions = []

            for selection_function, selection_function_result in results.items():

                mu = []
                fitnesses = []

                # get mu from the best fitness recording of the first run
                for m in selection_function_result[0]['best_fitnesses'].keys():
                    mu.append(m)
                    fitnesses.append(statistics.mean(r['best_fitnesses'][m] for r in selection_function_result))

                writer.writerow(mu)
                writer.writerow(fitnesses)

                selection_functions.append(selection_function)

        params = ['python3', 'plot_results.py', filename, self.results_directory, symlog]
        params.extend(selection_functions)
        print(params)
        subprocess.run(params)

        os.remove(filename)

    class popi:
        def __init__(self, other=None):
            if other is None:
                self.genome = None
                self.genome_type = None
                self.genome_length = None
                self.fitness = None
            else:
                self.genome = list(other.genome)
                self.genome_type = other.genome_type
                self.genome_length = other.genome_length

        def randomize(self, n, max_range, genome_type):
            self.genome = list()
            self.genome_length = n

            if genome_type == 'bool':
                self.genome_type = 'bool'
                for i in range(n):
                    self.genome.append(bool(random.random() > 0.5))
            elif genome_type == 'float':
                self.genome_type = 'float'
                for i in range(n):
                    self.genome.append(random.uniform(-max_range, max_range))

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
            new_child = basicEA.popi(self)
            if self.genome_type == 'bool':
                for i in range(self.genome_length):
                    if random.random() > 0.5:
                        new_child.genome[i] = parent2.genome[i]
            elif self.genome_type == 'float':
                for i in range(self.genome_length):
                    a = random.random()
                    new_child.genome[i] = a*self.genome[i] + (1-a)*parent2.genome[i]

            return new_child

    def evaluate_child(self, popi):
        if self.fitness_function == 'rosenbrock':
            popi.fitness = self.rosenbrock(popi.genome, self.fitness_function_a)
        elif self.fitness_function == 'rastrigin':
            popi.fitness = self.offset_rastrigin(popi.genome, self.fitness_function_a, self.fitness_function_offset)
        elif self.fitness_function == 'dtrap':
            popi.fitness = self.dtrap(popi.genome, self.trap_size)
        elif self.fitness_function == 'nk_landscape':
            popi.fitness = self.nk_landscape(popi.genome)

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
            locus.extend(list(x[j] for j in self.epistasis[i]))
            locus_fitness = self.loci_values[tuple(locus)]
            result += locus_fitness

        return result


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
            loci_values[locus] = random.randrange(0, n)

        epistasis = dict()
        for i in range(n):
            epistasis[i] = sorted(random.sample(list(j for j in range(n) if j != i), k))

        return loci_values, epistasis

    def parent_selection_eppsea_function(self, population, eppsea_selection_function, n, generation_number):
        return eppsea_selection_function.select(population, n, generation_number)

    def parent_selection_basic(self, population, unique_parents, selection_function):
        if selection_function == 'truncation':
            return max(unique_parents, key=lambda x: x.fitness)
        elif selection_function == 'fitness_rank':
            population.sort(key=lambda x: x.fitness, reverse=True)
            ranks = list(range(len(population), 0, -1))
            n = random.randint(0, sum(ranks))
            i = 0
            while n > ranks[i]:
                n -= ranks[i]
                i += 1
            return population[i]

        elif selection_function == 'fitness_proportional':
            minFitness = min(p.fitness for p in population)
            if minFitness < 0:
                selection_chances = [p.fitness - minFitness for p in population]
            else:
                selection_chances = [p.fitness for p in population]
            n = random.uniform(0, sum(selection_chances))
            i = 0
            while n > selection_chances[i]:
                n -= selection_chances[i]
                i += 1
            return population[i]

        elif selection_function == 'k_tournament':
            tournament = random.sample(population, self.tournament_k)
            winner = max(tournament, key=lambda p: p.fitness)
            return winner

        elif selection_function == 'random':
            return random.choice(population)

        else:
            print('PARENT SELECTION {0} NOT FOUND'.format(selection_function))

    def one_run(self, parent_selection_function, eppsea_selection_function):

        population = list()
        for i in range(self.mu):
            new_child = self.popi()
            new_child.randomize(self.genome_length, self.max_initial_range, self.genome_type)
            self.evaluate_child(new_child)
            population.append(new_child)

        evals = self.mu

        average_fitnesses = dict()
        average_fitnesses[evals] = statistics.mean(p.fitness for p in population)

        best_fitnesses = dict()
        best_fitnesses[evals] = max(p.fitness for p in population)

        generation_number = 0

        while evals <= self.max_evals:
            if parent_selection_function == 'eppsea_selection_function':
                children = []
                all_parents = self.parent_selection_eppsea_function(population, eppsea_selection_function, self.lam*2, generation_number)
                for i in range(0, len(all_parents), 2):
                    parent1 = all_parents[i]
                    parent2 = all_parents[i+1]

                    new_child = parent1.recombine(parent2)
                    for i in range(self.genome_length):
                        if random.random() < self.mutation_rate:
                            new_child.mutate_gene(i)

                    self.evaluate_child(new_child)

                    children.append(new_child)

                    evals += 1

            else:
                children = list()
                unique_parents = list(population)
                for i in range(self.lam):
                    parent1 = self.parent_selection_basic(population, unique_parents, parent_selection_function)
                    try:
                        unique_parents.remove(parent1)
                    except ValueError:
                        pass
                    parent2 = self.parent_selection_basic(population, unique_parents, parent_selection_function)
                    try:
                        unique_parents.remove(parent2)
                    except ValueError:
                        pass

                    new_child = parent1.recombine(parent2)
                    for i in range(self.genome_length):
                        if random.random() < self.mutation_rate:
                            new_child.mutate_gene(i)

                    self.evaluate_child(new_child)

                    children.append(new_child)

                    evals += 1

            population.extend(children)
            population.sort(key=lambda p: p.fitness)
            newPopulation = []
            newPopulation.append(population.pop())
            newPopulation.extend(random.sample(population, self.mu-1))
            population = newPopulation

            average_fitnesses[evals] = statistics.mean(p.fitness for p in population)

            best_fitnesses[evals] = max(p.fitness for p in population)

            generation_number += 1

        results = dict()
        results['final_average_fitness'] = statistics.mean(p.fitness for p in population)
        results['final_best_fitness'] = max(p.fitness for p in population)
        results['final_fitness_std_dev'] = statistics.stdev(p.fitness for p in population)
        results['average_fitnesses'] = average_fitnesses
        results['best_fitnesses'] = best_fitnesses


        return results

def main(config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = basicEA(config)
    shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))

    print('Now starting EPPSEA')
    start_time = time.time()

    eppsea_config = config.get('EA', 'base eppsea config path')
    eppsea = eppsea_base.Eppsea(eppsea_config)

    eppsea.start_evolution()

    using_multiprocessing = config.getboolean('EA', 'use multiprocessing')

    while not eppsea.evolution_finished:
        evaluate_eppsea_population(evaluator, eppsea.new_population, using_multiprocessing)
        eppsea.next_generation()

    best_selection_function = eppsea.final_best_member
    final_results = test_against_basic_selection(evaluator, best_selection_function)
    end_time = time.time() - start_time
    evaluator.log('Time elapsed: {0}'.format(end_time))

    final_results_path = '{0}/final_results'.format(evaluator.results_directory)
    with open(final_results_path, 'wb') as pickle_file:
        pickle.dump(final_results, pickle_file)

    try:
        postprocess_results = postprocess(final_results_path, evaluator.results_directory)
        evaluator.log('Postprocess results:')
        evaluator.log(postprocess_results)
    except Exception as e:
        evaluator.log('Postprocessing failed. Run postprocessing directly on {0}'.format(final_results_path))

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    main(config_path)
