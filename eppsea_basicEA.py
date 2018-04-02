import math
import random
import statistics
import itertools
import sys
import configparser
import os
import shutil
import subprocess
import multiprocessing
import time
import datetime
import pickle

import eppsea_base
import find_optimal_tournament_k


def postprocess(final_results_path, results_directory):
    # calls the postprocessing script on a pickled dictionary mapping fitness functions to ResultHolder objects
    params = ['python3', 'post_process.py', final_results_path, results_directory]
    result = subprocess.run(params, stdout=subprocess.PIPE, universal_newlines=True)

    return result.stdout

def run_ea(ea):
    result = ea.one_run()
    return result

class ResultHolder:
    # a class for holding the results of eppsea runs
    def __init__(self):
        self.selection_function_name = None
        self.fitness_function_name = None
        self.run_results = []

    def get_eval_counts(self):
        all_counts = set()
        for r in self.run_results:
            counts = r['best_fitnesses'].keys()
            all_counts.update(counts)
        return sorted(list(all_counts))

    def get_average_average_fitness(self):
        average_average_fitnesses = []
        for m in self.get_eval_counts():
            average_fitnesses = []
            for r in self.run_results:
                if m in r['average_fitnesses']:
                    average_fitnesses.append(r['average_fitnesses'][m])
                # if this run terminated early, use the final average fitness
                elif all(m > rm for rm in r['average_fitnesses'].keys()):
                    average_fitnesses.append(r['final_average_fitness'])
            average_average_fitness = statistics.mean(average_fitnesses)
            average_average_fitnesses.append(average_average_fitness)

        return average_average_fitnesses

    def get_average_best_fitness(self):
        average_best_fitnesses = []
        for m in self.get_eval_counts():
            best_fitnesses = []
            for r in self.run_results:
                if m in r['best_fitnesses']:
                    best_fitnesses.append(r['best_fitnesses'][m])
                # if this run terminated early, use the final best fitness
                elif all(m > rm for rm in r['best_fitnesses'].keys()):
                    best_fitnesses.append(r['final_best_fitness'])
            average_best_fitness = statistics.mean(best_fitnesses)
            average_best_fitnesses.append(average_best_fitness)

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


class FitnessFunction:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool'
    }

    def __init__(self, config):

        self.fitness_function_name = config.get('EA', 'fitness function')
        self.genome_type = self.genome_types.get(self.fitness_function_name, None)

        self.genome_length = config.getint('fitness function', 'genome length')
        self.fitness_function_a = config.getfloat('fitness function', 'a')
        self.max_initial_range = config.getfloat('fitness function', 'max initial range')
        self.trap_size = config.getint('fitness function', 'trap size')
        self.epistasis_k = config.getint('fitness function', 'epistasis k')

        if self.fitness_function_name == 'rastrigin':
            self.fitness_function_offset = self.generate_offset(self.genome_length)
        else:
            self.fitness_function_offset = None

        if self.fitness_function_name == 'nk_landscape':
            self.loci_values, self.epistasis = self.generate_epistatis(self.genome_length, self.epistasis_k)
        else:
            self.loci_values, self.epistasis = None, None

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

    def evaluate(self, genome):
        if self.fitness_function_name == 'rosenbrock':
            fitness = self.rosenbrock(genome, self.fitness_function_a)
        elif self.fitness_function_name == 'rastrigin':
            fitness = self.offset_rastrigin(genome, self.fitness_function_a, self.fitness_function_offset)
        elif self.fitness_function_name == 'dtrap':
            fitness = self.dtrap(genome, self.trap_size)
        elif self.fitness_function_name == 'nk_landscape':
            fitness = self.nk_landscape(genome)

        else:
            raise Exception('EPPSEA BasicEA ERROR: fitness function name {0} not recognized'.format(self.fitness_function_name))

        return fitness


class EA:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool'
    }

    def __init__(self, config, fitness_function, selection_function_name, selection_function, tournament_k):
        self.mu = config.getint('EA', 'population size')
        self.lam = config.getint('EA', 'offspring size')

        self.mutation_rate = config.getfloat('EA', 'mutation rate')
        self.max_evals = config.getint('EA', 'maximum evaluations')
        self.convergence_termination = config.getboolean('EA', 'terminate on convergence')
        self.convergence_generations = config.getint('EA', 'generations to convergence')
        self.target_termination = config.getboolean('EA', 'terminate at target fitness')
        self.target_fitness = config.getfloat('EA', 'target fitness')
        self.survival_selection = config.get('EA', 'survival selection')
        self.tournament_k = tournament_k

        self.fitness_function_name = config.get('EA', 'fitness function')
        self.fitness_function = fitness_function

        self.selection_function_name = selection_function_name
        self.selection_function = selection_function

    class Popi:
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
            new_child = EA.Popi(self)
            if self.genome_type == 'bool':
                for i in range(self.genome_length):
                    if random.random() > 0.5:
                        new_child.genome[i] = parent2.genome[i]
            elif self.genome_type == 'float':
                for i in range(self.genome_length):
                    a = random.random()
                    new_child.genome[i] = a*self.genome[i] + (1-a)*parent2.genome[i]

            return new_child

    def evaluate_child(self, popi, fitness_function):
        popi.fitness = fitness_function.evaluate(popi.genome)

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
            min_fitness = min(p.fitness for p in population)
            if min_fitness < 0:
                selection_chances = [p.fitness - min_fitness for p in population]
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

    def one_run(self):
        generation_number = 0

        population = list()
        for _ in range(self.mu):
            new_child = self.Popi()
            new_child.randomize(self.fitness_function.genome_length, self.fitness_function.max_initial_range, self.fitness_function.genome_type)
            new_child.birth_gen = generation_number
            self.evaluate_child(new_child, self.fitness_function)
            population.append(new_child)

        evals = self.mu

        average_fitnesses = dict()
        average_fitnesses[evals] = statistics.mean(p.fitness for p in population)

        best_fitnesses = dict()
        best_fitnesses[evals] = max(p.fitness for p in population)

        generations_since_best_fitness_improvement = 0
        previous_best_fitness = -math.inf

        generation_number = 1

        while evals <= self.max_evals:
            if self.selection_function_name == 'eppsea_selection_function':
                children = []
                all_parents = self.parent_selection_eppsea_function(population, self.selection_function, self.lam*2, generation_number)
                for i in range(0, len(all_parents), 2):
                    parent1 = all_parents[i]
                    parent2 = all_parents[i+1]

                    new_child = parent1.recombine(parent2)
                    new_child.birth_gen = generation_number
                    for j in range(new_child.genome_length):
                        if random.random() < self.mutation_rate:
                            new_child.mutate_gene(j)

                    self.evaluate_child(new_child, self.fitness_function)

                    children.append(new_child)

                    evals += 1

            else:
                children = list()
                unique_parents = list(population)
                for _ in range(self.lam):
                    parent1 = self.parent_selection_basic(population, unique_parents, self.selection_function_name)
                    try:
                        unique_parents.remove(parent1)
                    except ValueError:
                        pass
                    parent2 = self.parent_selection_basic(population, unique_parents, self.selection_function_name)
                    try:
                        unique_parents.remove(parent2)
                    except ValueError:
                        pass

                    new_child = parent1.recombine(parent2)
                    new_child.birth_gen = generation_number
                    for i in range(new_child.genome_length):
                        if random.random() < self.mutation_rate:
                            new_child.mutate_gene(i)

                    self.evaluate_child(new_child, self.fitness_function)

                    children.append(new_child)

                    evals += 1

            population.extend(children)

            if self.target_termination and any(p.fitness >= self.target_fitness for p in population):
                break

            if self.survival_selection == 'random':
                population = random.sample(population, self.mu)
            elif self.survival_selection == 'elitist_random':
                population.sort(key=lambda p: p.fitness, reverse=True)
                new_population = []
                new_population.append(population.pop(0))
                new_population.extend(random.sample(population, self.mu-1))
                population = new_population
            elif self.survival_selection == 'truncation':
                population.sort(key=lambda p: p.fitness, reverse=True)
                population = population[:self.mu]
            else:
                raise Exception('ERROR: Configuration parameter for survival selection {0} not recognized'.format(self.survival_selection))

            average_fitness = statistics.mean(p.fitness for p in population)
            average_fitnesses[evals] = average_fitness

            best_fitness = max(p.fitness for p in population)
            best_fitnesses[evals] = best_fitness

            generation_number += 1

            if best_fitness > previous_best_fitness:
                previous_best_fitness = best_fitness
                generations_since_best_fitness_improvement = 0
            else:
                generations_since_best_fitness_improvement += 1
                if self.convergence_termination and generations_since_best_fitness_improvement >= self.convergence_generations:
                    break

        run_results = dict()
        run_results['final_average_fitness'] = statistics.mean(p.fitness for p in population)
        run_results['final_best_fitness'] = max(p.fitness for p in population)
        run_results['final_fitness_std_dev'] = statistics.stdev(p.fitness for p in population)
        run_results['average_fitnesses'] = average_fitnesses
        run_results['best_fitnesses'] = best_fitnesses

        return run_results


class EppseaBasicEA:
    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool',
        'nk_landscape': 'bool'
    }

    def __init__(self, config):
        self.config = config

        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_basicEA_" + str(present_time)

        self.results_directory = 'results/eppsea_basicEA/{0}'.format(experiment_name)
        os.makedirs(self.results_directory, exist_ok=True)

        self.log_file_location = '{0}/log.txt'.format(self.results_directory)

        self.using_multiprocessing = config.getboolean('EA', 'use multiprocessing')

        self.training_runs = config.getint('EA', 'training runs')
        self.testing_runs = config.getint('EA', 'testing runs')

        self.fitness_function_policy = config.get('EA', 'fitness function policy')
        if self.fitness_function_policy == 'single':
            self.training_fitness_functions = [FitnessFunction(config)]
            self.testing_fitness_functions = []
        elif self.fitness_function_policy == 'multiple':
            self.training_fitness_functions = []
            self.testing_fitness_functions = []
            for _ in range(config.getint('EA', 'num training fitness functions')):
                self.training_fitness_functions.append(FitnessFunction(config))
        elif self.fitness_function_policy == 'generalization':
            self.training_fitness_functions = []
            self.testing_fitness_functions = []
            for _ in range(config.getint('EA', 'num training fitness functions')):
                self.training_fitness_functions.append(FitnessFunction(config))
            for _ in range(config.getint('EA', 'num testing fitness functions')):
                self.testing_fitness_functions.append(FitnessFunction(config))
        else:
            raise Exception('EPPSEA BasicEA ERROR: fitness function policy {0} not recognized'.format(self.fitness_function_policy))

        self.num_training_fitness_functions = len(self.training_fitness_functions)
        self.num_testing_fitness_functions = len(self.testing_fitness_functions)

        try:
            self.tournament_k = config.getint('EA', 'tournament k')
        except ValueError:
            self.log('Determining optimal K for K tournament...')
            self.tournament_k = None
            optimal_k = find_optimal_tournament_k.find_optimal_k(self)
            self.log('Optimal K value is {0}'.format(optimal_k))
            self.tournament_k = optimal_k

        self.basic_results = None

    def log(self, message):
        print(message)
        with open(self.log_file_location, 'a') as log_file:
            log_file.write(message + '\n')

    def get_eas(self, selection_functions, is_testing):
        # this prepares and returns a list of EAs for the provided selection functions
        # selection_functions is a list of tuples of the form (selection_function_name, selection_function),
        # where, if the selection_function_name is 'eppsea_selection_function', then selection_function is
        # expected to be an eppsea selection function

        result = list()

        for selection_function_name, selection_function in selection_functions:

            ###sanity check
            if selection_function_name == 'eppsea_selection_function':
                assert(selection_function is not None)
            else:
                assert(selection_function is None)

            if is_testing:
                for f in self.testing_fitness_functions:
                    result.append(EA(self.config, f, selection_function_name, selection_function, self.tournament_k))
            else:
                for f in self.training_fitness_functions:
                    result.append(EA(self.config, f, selection_function_name, selection_function, self.tournament_k))

        return result

    def run_eas(self, eas, is_testing):
        # runs each of the eas for the configured number of runs, returning one list for each ea
        results_all_eas = []

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

                result_holder = ResultHolder()
                result_holder.selection_function_name = eas[i].selection_function_name
                result_holder.fitness_function_name = eas[i].fitness_function_name
                result_holder.run_results = results[start:stop]

                results_all_eas.append(result_holder)

        else:
            for ea in eas:
                results = []
                for r in range(runs):
                    results.append(run_ea(ea))

                result_holder = ResultHolder()
                result_holder.selection_function_name = ea.selection_function_name
                result_holder.fitness_function_name = ea.fitness_function_name
                result_holder.run_results = results

        return results_all_eas

    def evaluate_eppsea_population(self, eppsea_population, is_testing):
        # evaluates a population of eppsea individuals and assigns fitness values to them
        if is_testing:
            num_fitness_functions = self.num_testing_fitness_functions
        else:
            num_fitness_functions = self.num_training_fitness_functions

        selection_functions = list(('eppsea_selection_function', e) for e in eppsea_population)
        eas = self.get_eas(selection_functions, is_testing)
        ea_results = self.run_eas(eas, is_testing)
        for i in range(len(eppsea_population)):
            start = i * num_fitness_functions
            stop = (i + 1) * num_fitness_functions
            fitnesses = (r.get_average_final_best_fitness() for r in ea_results[start:stop])
            fitness = statistics.mean(fitnesses)
            eppsea_population[i].fitness = fitness

    def test_against_basic_selection(self, eppsea_selection_function, is_testing):
        selection_function_names = ['truncation', 'fitness_proportional', 'fitness_rank', 'k_tournament', 'random']
        if self.config.getint('EA', 'offspring size') > self.config.getint('EA', 'population size') * 2:
            selection_function_names.remove('truncation')

        selection_functions = []
        for s in selection_function_names:
            selection_functions.append((s, None))
        selection_functions.append(('eppsea_selection_function', eppsea_selection_function))

        eas = self.get_eas(selection_functions, is_testing)

        ea_results = self.run_eas(eas, is_testing)

        return ea_results

    def run_eppsea_basicea(self):
        print('Now starting EPPSEA')
        start_time = time.time()

        eppsea_config = self.config.get('EA', 'base eppsea config path')
        eppsea = eppsea_base.Eppsea(eppsea_config)

        eppsea.start_evolution()

        while not eppsea.evolution_finished:
            self.evaluate_eppsea_population(eppsea.new_population, False)
            eppsea.next_generation()

        best_selection_function = eppsea.final_best_member
        final_test_results = self.test_against_basic_selection(best_selection_function, True)
        end_time = time.time() - start_time
        self.log('Time elapsed: {0}'.format(end_time))

        all_fitness_function_results = dict()
        for i in range(self.num_testing_fitness_functions):
            fitness_function_results = final_test_results[i::self.num_testing_fitness_functions]
            all_fitness_function_results[i] = fitness_function_results

        final_results_path = '{0}/final_results'.format(self.results_directory)
        with open(final_results_path, 'wb') as pickle_file:
            pickle.dump(all_fitness_function_results, pickle_file)

        try:
            postprocess_results = postprocess(final_results_path, self.results_directory)
            self.log('Postprocess results:')
            self.log(postprocess_results)
        except Exception:
            self.log('Postprocessing failed. Run postprocessing directly on {0}'.format(final_results_path))

        eppsea_base_results_path = eppsea.results_directory
        shutil.copytree(eppsea_base_results_path, self.results_directory + '/base')


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = EppseaBasicEA(config)
    shutil.copy(config_path, '{0}/config.cfg'.format(evaluator.results_directory))
    evaluator.run_eppsea_basicea()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    main(config_path)
