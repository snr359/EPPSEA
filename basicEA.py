import math
import random
import statistics

class popi:
    def __init__(self):
        self.genome = None
        self.genome_type = None
        self.genome_length = None
        self.fitness = None

    def randomize(self, n, max_range, genome_type):
        self.genome = list()
        self.genome_length = n

        if genome_type == 'bool':
            self.genome_type = 'bool'
            for i in range(n):
                self.genome.append(bool(random.random() > 0.5))
        elif genome_type == 'real':
            self.genome_type = 'real'
            for i in range(n):
                self.genome.append(random.uniform(-max_range, max_range))

    def mutate_gene(self, gene):
        if self.genome_type == 'bool':
            self.genome[gene] = not self.genome[gene]

        elif self.genome_type == 'real':
            self.genome[gene] += random.triangular(-1, 1, 0)

    def mutate_one(self):
        gene = random.randrange(self.genome_length)
        self.mutate_gene(gene)

    def mutate_all(self):
        for i in range(self.genome_length):
            self.mutate_gene(i)

    def recombine(self, parent2):
        new_child = popi()
        new_child.genome = list(self.genome)
        new_child.genome_length = self.genome_length
        new_child.genome_type = self.genome_type
        for i in range(self.genome_length):
            if random.random() > 0.5:
                new_child.genome[i] = parent2.genome[i]

        return new_child

    def evaluate(self, fitness_function, fitness_function_offset):
        if fitness_function == 'rosenbrock':
            self.fitness = rosenbrock(self.genome, 100)
        elif fitness_function == 'rastrigin':
            self.fitness = offset_rastrigin(self.genome, 10, fitness_function_offset)

class testSelectionFunction:
    def __init__(self):
        self.reusingParents = False

    def get(self, terminals):
        return terminals['fitness rank']

def rosenbrock(x, a):
    result = 0
    n = len(x)
    for i in range(n-1):
        if -5 <= x[i] <= 10:
            result += (1 - x[i])**2 + a*(x[i+1] - x[i]**2)**2

    result = -1 * result
    return result

def rastrigin(x, a):
    n = len(x)
    result = a * n
    for i in range(n):
        result += x[i]**2 - a*math.cos(2*math.pi*x[i])

    result = -1 * result
    return result

def offset_rastrigin(x, a, offset):
    offset_x = list(x)
    for i in range(len(x)):
        offset_x[i] += offset[i]
    return rastrigin(offset_x, a)

def generate_offset(n):
    result = []
    for _ in range(n):
        result.append(random.choice([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    return result

def set_fitness_stats(population):
    sortedPopulation = sorted(population, key=lambda p:p.fitness)
    for i, p in enumerate(sortedPopulation):
        p.fitness_rank = i

    totalFitness = float(sum(p.fitness for p in population))
    for p in population:
        p.fitness_proportion = p.fitness / totalFitness

def set_selection_chances(population, selection_function):
    for p in population:
        terminals = dict()
        terminals['fitness'] = p.fitness
        terminals['fitnessProportion'] = p.fitness_proportion
        terminals['fitnessRank'] = p.fitness_rank
        terminals['populationSize'] = len(population)
        p.selection_chance = selection_function.get(terminals)

    min_chance = min(p.selection_chance for p in population)
    if min_chance < 0:
        for p in population:
            p.selection_chance -= min_chance

def parent_selection(population, parents_used, selection_function):
    candidates = list(population)
    if not selection_function.reusingParents:
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

    if selected == None:
        raise Exception('Overran selection function with {0} remaining'.format(selection_num))

    return selected

def one_run(fitness_function, genome_length, max_evals, population_size, offspring_size, mutation_rate, selection_function):
    if fitness_function == 'rasrigin':
        fitness_function_offset = generate_offset(genome_length)
    else:
        fitness_function_offset = None

    if fitness_function in ['rosenbrock',
                            'rastrigin']:
        genome_type = 'real'
        max_range = 5
    else:
        #TODO: boolean fitness functions
        genome_type = 'bool'
        max_range = 1

    population = list()
    for i in range(population_size):
        new_child = popi()
        new_child.randomize(genome_length, max_range, genome_type)
        new_child.evaluate(fitness_function, fitness_function_offset)
        population.append(new_child)

    evals = population_size

    while evals <= max_evals:
        children = list()
        parents_used = list()
        for i in range(offspring_size):
            set_fitness_stats(population)
            set_selection_chances(population, selection_function)
            parent1 = parent_selection(population, parents_used, selection_function)
            parents_used.append(parent1)
            parent2 = parent_selection(population, parents_used, selection_function)
            parents_used.append(parent2)

            new_child = parent1.recombine(parent2)
            if random.random() < mutation_rate:
                new_child.mutate_all()

            new_child.evaluate(fitness_function, fitness_function_offset)

            children.append(new_child)

            evals += 1

        population.extend(children)
        population.sort(key=lambda p: p.fitness, reverse=True)
        population = random.sample(population, population_size)

    results = dict()
    results['average_fitness'] = statistics.mean(p.fitness for p in population)
    results['best_fitness'] = max(p.fitness for p in population)

    return results

if __name__ == '__main__':
    test_selection_function = testSelectionFunction()
    print(one_run('rosenbrock', 20, 5000, 50, 50, 0.1, test_selection_function))

