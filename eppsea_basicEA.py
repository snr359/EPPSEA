import math
import random
import statistics
import itertools
import sys
import configparser

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

        if self.fitness_function == 'rastrigin':
            self.fitness_function_offset = self.generate_offset(self.genome_length)
        else:
            self.fitness_function_offset = None

        if self.fitness_function == 'nk_landscape':
            self.loci_values, self.epistasis = self.generate_epistatis(self.genome_length, self.epistasis_k)
        else:
            self.loci_values, self.epistasis = None, None

    def evaluate(self, eppsea_selection_function):
        results = list()
        for _ in range(self.runs):
            results.append(self.one_run(eppsea_selection_function))

        average_best = statistics.mean(r['best_fitness'] for r in results)

        return average_best

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
                self.genome[gene] += random.triangular(-1, 1, 0)

        def mutate_one(self):
            gene = random.randrange(self.genome_length)
            self.mutate_gene(gene)

        def mutate_all(self):
            for i in range(self.genome_length):
                self.mutate_gene(i)

        def recombine(self, parent2):
            new_child = basicEA.popi(self)
            for i in range(self.genome_length):
                if random.random() > 0.5:
                    new_child.genome[i] = parent2.genome[i]

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

        totalFitness = float(sum(p.fitness for p in population))
        for p in population:
            p.fitness_proportion = p.fitness / totalFitness

    def set_selection_chances(self, population, eppsea_selection_function):
        for p in population:
            terminals = dict()
            terminals['fitness'] = p.fitness
            terminals['fitnessProportion'] = p.fitness_proportion
            terminals['fitnessRank'] = p.fitness_rank
            terminals['populationSize'] = len(population)
            p.selection_chance = eppsea_selection_function.get(terminals)

        min_chance = min(p.selection_chance for p in population)
        if min_chance < 0:
            for p in population:
                p.selection_chance -= min_chance

    def parent_selection_eppsea_function(self, population, parents_used, eppsea_selection_function):
        candidates = list(population)
        if not eppsea_selection_function.reusingParents:
            for p in parents_used:
                try:
                    candidates.remove(p)
                except ValueError:
                    pass

        if len(candidates) == 0:
            raise Exception('Trying to select candidate for parent selection from empty pool!')

        total_chance = sum(p.selection_chance for p in candidates)
        selection_num = random.uniform(0, total_chance)
        selected = None
        for p in population:
            if selection_num <= p.selection_chance:
                selected = p
                break
            else:
                selection_num -= p.selection_chance

        if selected is None:
            raise Exception('Overran selection function with {0} remaining'.format(selection_num))

        return selected

    def one_run(self, eppsea_selection_function):

        population = list()
        for i in range(self.mu):
            new_child = self.popi()
            new_child.randomize(self.genome_length, self.max_initial_range, self.genome_type)
            self.evaluate_child(new_child)
            population.append(new_child)

        evals = self.mu

        while evals <= self.max_evals:
            children = list()
            parents_used = list()
            for i in range(self.lam):
                self.set_fitness_stats(population)
                self.set_selection_chances(population, eppsea_selection_function)
                parent1 = self.parent_selection_eppsea_function(population, parents_used, eppsea_selection_function)
                parents_used.append(parent1)
                parent2 = self.parent_selection_eppsea_function(population, parents_used, eppsea_selection_function)
                parents_used.append(parent2)

                new_child = parent1.recombine(parent2)
                if random.random() < self.mutation_rate:
                    new_child.mutate_all()

                self.evaluate_child(new_child)

                children.append(new_child)

                evals += 1

            population.extend(children)
            population.sort(key=lambda p: p.fitness, reverse=True)
            population = random.sample(population, self.mu)

        results = dict()
        results['average_fitness'] = statistics.mean(p.fitness for p in population)
        results['best_fitness'] = max(p.fitness for p in population)

        return results

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide config file')
        exit(1)

    config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    evaluator = basicEA(config)
    eppsea_base.eppsea(evaluator, 'config/base_config/basicEA.cfg')
