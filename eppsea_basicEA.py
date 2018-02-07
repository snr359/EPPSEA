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

import eppsea_base

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

    def evaluate(self, eppsea_selection_function):
        if self.lam > self.mu / 2:
            eppsea_selection_function.reusingParents = True
        results = list()
        for _ in range(self.runs):
            results.append(self.one_run('eppsea_selection_function', eppsea_selection_function))

        average_best = statistics.mean(r['best_fitness'] for r in results)
        std_dev_best = statistics.stdev(r['best_fitness'] for r in results)

        if eppsea_selection_function.final:
            for basic_selection, basic_selection_results in self.basic_results.items():
                average_best_difference = average_best - statistics.mean(r['best_fitness'] for r in basic_selection_results)
                with open('temp.csv', 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(list(r['best_fitness'] for r in results))
                    writer.writerow(list(r['best_fitness'] for r in basic_selection_results))

                t_test_results = subprocess.check_output(['python3', 't_test.py', 'temp.csv'])
                _, _, t, p_value = list(float(r) for r in t_test_results.split())
                print('Compared to selection function {0}, difference in average fitness is {1}. P-value: {2}'.format(basic_selection, average_best_difference, p_value))

        return {'fitness': average_best,
                'std_dev': std_dev_best}

    def test_basic_selection(self):
        self.basic_results = dict()
        with open('basicEA_results/basicEA_{0}_basic_selection.log'.format(self.fitness_function), 'w') as log_file:
            for parent_selection_function in ['truncation',
                                              'fitness_proportional',
                                              'fitness_rank',
                                              'k_tournament']:
                if parent_selection_function == 'truncation' and self.lam > self.mu / 2:
                    continue
                results = list()
                for _ in range(self.runs):
                    results.append(self.one_run(parent_selection_function))

                average_best = statistics.mean(r['best_fitness'] for r in results)
                std_dev_best = statistics.stdev(r['best_fitness'] for r in results)

                log_file.write('Average average fitness and standard deviation for fitness function {0} using selection function {1}: {2}, {3}\n'.format(
                    self.fitness_function, parent_selection_function, average_best, std_dev_best))

                self.basic_results[parent_selection_function] = results


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

    def set_fitness_stats(self, population):
        sortedPopulation = sorted(population, key=lambda p:p.fitness)
        for i, p in enumerate(sortedPopulation):
            p.fitness_rank = i

    def set_selection_chances(self, population, eppsea_selection_function):
        terminals = dict()
        terminals['populationSize'] = len(population)
        terminals['sumFitness'] = sum(p.fitness for p in population)
        for p in population:
            terminals['fitness'] = p.fitness
            terminals['fitnessRank'] = p.fitness_rank
            p.selection_chance = eppsea_selection_function.get(terminals)

        min_chance = min(p.selection_chance for p in population)
        if min_chance < 0:
            for p in population:
                p.selection_chance -= min_chance

    def parent_selection_eppsea_function(self, population, unique_parents, eppsea_selection_function):
        if not eppsea_selection_function.reusingParents:
            candidates = list(unique_parents)
        else:
            candidates = list(population)

        if len(candidates) == 0:
            raise Exception('Trying to select candidate for parent selection from empty pool!')

        total_chance = sum(p.selection_chance for p in candidates)
        selection_num = random.uniform(0, total_chance)
        selected = None
        for p in candidates:
            if selection_num <= p.selection_chance:
                selected = p
                break
            else:
                selection_num -= p.selection_chance

        if selected is None:
            raise Exception('Overran selection function with {0} remaining'.format(selection_num))

        return selected

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

    def one_run(self, parent_selection_function, eppsea_selection_function=None):

        population = list()
        for i in range(self.mu):
            new_child = self.popi()
            new_child.randomize(self.genome_length, self.max_initial_range, self.genome_type)
            self.evaluate_child(new_child)
            population.append(new_child)

        evals = self.mu

        average_fitnesses = dict()
        average_fitnesses[self.mu] = statistics.mean(p.fitness for p in population)

        best_fitnesses = dict()
        best_fitnesses[self.mu] = max(p.fitness for p in population)

        while evals <= self.max_evals:
            children = list()
            unique_parents = list(population)
            if parent_selection_function == 'eppsea_selection_function':
                self.set_fitness_stats(population)
                self.set_selection_chances(population, eppsea_selection_function)
            for i in range(self.lam):
                if parent_selection_function == 'eppsea_selection_function':
                    parent1 = self.parent_selection_eppsea_function(population, unique_parents, eppsea_selection_function)
                    try:
                        unique_parents.remove(parent1)
                    except ValueError:
                        pass
                    parent2 = self.parent_selection_eppsea_function(population, unique_parents, eppsea_selection_function)
                    try:
                        unique_parents.remove(parent2)
                    except ValueError:
                        pass
                else:
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

            average_fitnesses[self.mu] = statistics.mean(p.fitness for p in population)

            best_fitnesses[self.mu] = max(p.fitness for p in population)

        results = dict()
        results['average_fitness'] = statistics.mean(p.fitness for p in population)
        results['best_fitness'] = max(p.fitness for p in population)
        results['fitness_std_dev'] = statistics.stdev(p.fitness for p in population)
        results['average_fitnesses'] = average_fitnesses
        results['best_fitnesses'] = best_fitnesses

        return results

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    os.makedirs('basicEA_results', exist_ok=True)
    shutil.copy(config_path, 'basicEA_results/config.cfg')

    evaluator = basicEA(config)
    print('First, testing basic selection functions')
    evaluator.test_basic_selection()
    print('Now starting EPPSEA')
    base_eppsea_config_path = config.get('EA', 'base eppsea config path')
    eppsea_base.eppsea(evaluator, base_eppsea_config_path)

