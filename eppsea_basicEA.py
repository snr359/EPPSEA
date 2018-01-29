import math
import random
import statistics

import eppsea_base

class basicEA:

    genome_lengths = {
        'rosenbrock': 20,
        'rastrigin': 20,
        'dtrap': 400
    }

    fitness_function_as = {
        'rosenbrock': 100,
        'rastrigin': 10,
    }

    genome_types = {
        'rastrigin': 'float',
        'rosenbrock': 'float',
        'dtrap': 'bool'
    }

    max_ranges = {
        'rosenbrock': 5,
        'rastrigin': 5
    }

    trap_sizes = {
        'dtrap': 4
    }

    def __init__(self, fitness_function, mu, lam, mutation_rate, max_evals, runs):
        self.fitness_function = fitness_function
        self.mu = mu
        self.lam = lam

        self.mutation_rate = mutation_rate
        self.max_evals = max_evals

        self.runs = runs

        if self.fitness_function == 'rastrigin':
            self.fitness_function_offset = self.generate_offset(20)
        else:
            self.fitness_function_offset = None

        self.genome_type = self.genome_types.get(fitness_function, None)
        self.genome_length = self.genome_lengths.get(fitness_function, None)
        self.fitness_function_a = self.fitness_function_as.get(fitness_function, None)
        self.max_range = self.max_ranges.get(fitness_function, None)
        self.trap_size = self.trap_sizes.get(fitness_function, None)

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
            new_child.randomize(self.genome_length, self.max_range, self.genome_type)
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
    fitness_function = 'dtrap'

    mu = 50
    lam = 20
    mutation_rate = 0.1

    max_evals = 5000
    runs = 2

    evaluator = basicEA(fitness_function, mu, lam, mutation_rate, max_evals, runs)
    eppsea_base.eppsea(evaluator, 'config/base_config/test.cfg')

