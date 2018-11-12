import time
import copy
import random
import datetime
import os
import csv
import math
import configparser
import shutil
import pickle
import statistics
import uuid
import numpy
import scipy.spatial


class GPNode:
    numeric_terminals = ['constant', 'random']
    data_terminals = ['fitness', 'fitness_rank', 'relative_fitness', 'birth_generation', 'relative_uniqueness', 'population_size', 'min_fitness', 'sum_fitness', 'max_fitness', 'generation_number']

    terminals = numeric_terminals + data_terminals

    non_terminals = ['+', '-', '*', '/', 'step', 'absolute', 'min', 'max']
    child_count = {'+': 2, '-': 2, '*': 2, '/': 2, 'step': 2, 'absolute': 1, 'min': 2, 'max': 2}

    def __init__(self, constant_min, constant_max, random_min, random_max):
        self.operation = None
        self.data = None
        self.children = None
        self.parent = None

        self.constant_min = constant_min
        self.constant_max = constant_max
        
        self.random_min = random_min
        self.random_max = random_max

    def grow(self, depth_limit, terminal_node_generation_chance, parent):
        if depth_limit == 0 or random.random() < terminal_node_generation_chance:
            self.operation = random.choice(GPNode.terminals)
        else:
            self.operation = random.choice(GPNode.non_terminals)

        if self.operation == 'constant':
            self.data = random.uniform(self.constant_min, self.constant_max)
        if self.operation in GPNode.non_terminals:
            self.children = []
            for i in range(GPNode.child_count[self.operation]):
                new_child_node = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
                new_child_node.grow(depth_limit - 1, terminal_node_generation_chance, self)
                self.children.append(new_child_node)
        self.parent = parent

    def get(self, terminal_values):
        if self.operation == '+':
            return self.children[0].get(terminal_values) + self.children[1].get(terminal_values)
        elif self.operation == '-':
            return self.children[0].get(terminal_values) - self.children[1].get(terminal_values)
        elif self.operation == '*':
            return self.children[0].get(terminal_values) * self.children[1].get(terminal_values)
        elif self.operation == '/':
            numerator = self.children[0].get(terminal_values)
            denominator = self.children[1].get(terminal_values)
            denominator = numpy.where(denominator!=0, denominator, 0.000001)
            return numpy.divide(numerator, denominator, where=denominator!=0)

        elif self.operation == 'step':
            return numpy.array(self.children[0].get(terminal_values) >= self.children[1].get(terminal_values), dtype=int)

        elif self.operation == 'absolute':
            return numpy.absolute(self.children[0].get(terminal_values))

        elif self.operation == 'min':
            return numpy.amin(numpy.stack((self.children[0].get(terminal_values), self.children[1].get(terminal_values))), axis=0)

        elif self.operation == 'max':
            return numpy.amax(numpy.stack((self.children[0].get(terminal_values), self.children[1].get(terminal_values))), axis=0)

        elif self.operation in GPNode.data_terminals:
            return terminal_values[self.operation]

        elif self.operation == 'constant':
            population_size = terminal_values['population_size'][0]
            return numpy.repeat(self.data, population_size)

        elif self.operation == 'random':
            population_size = terminal_values['population_size'][0]
            return numpy.random.uniform(self.random_min, self.random_max, population_size)

    def get_string(self):
        result = ''
        if self.operation in GPNode.non_terminals:
            if self.child_count[self.operation] == 2:
                result = "(" + self.children[0].get_string() + " " + self.operation + " " + self.children[1].get_string() + ")"
            elif self.child_count[self.operation] == 1:
                result = self.operation + "(" + self.children[0].get_string() + ")"
        elif self.operation == 'constant':
            result = str(self.data)
        else:
            result = self.operation
        return result

    def get_all_nodes(self):
        nodes = []
        nodes.append(self)
        if self.children is not None:
            for c in self.children:
                nodes.extend(c.get_all_nodes())
        return nodes

    def get_all_nodes_depth_limited(self, depth_limit):
        # returns all nodes down to a certain depth limit
        nodes = []
        nodes.append(self)
        if self.children is not None and depth_limit > 0:
            for c in self.children:
                nodes.extend(c.get_all_nodes_depth_limited(depth_limit - 1))
        return nodes

    def get_dict(self):
        # return a dictionary containing the operation, data, and children of the node
        result = dict()
        result['data'] = self.data
        result['operation'] = self.operation
        if self.children is not None:
            result['children'] = []
            for c in self.children:
                result['children'].append(c.get_dict())
        return result

    def build_from_dict(self, d):
        # builds a GPTree from a dictionary output by get_dict
        self.data = d['data']
        self.operation = d['operation']
        if 'children' in d.keys():
            self.children = []
            for c in d['children']:
                new_node = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
                new_node.build_from_dict(c)
                new_node.parent = self
                self.children.append(new_node)
        else:
            self.children = None


class GPTree:
    # encapsulates a tree made of GPNodes that determine probability of selection, as well as other options relating
    # to parent selection
    selection_types = ['proportional_replacement',
                       'proportional_no_replacement',
                       'tournament_replacement',
                       'tournament_no_replacement',
                       'truncation',
                       'stochastic_universal_sampling']

    replacement_selections = ['proportional_replacement',
                              'tournament_replacement']

    def __init__(self):
        self.root = None
        self.fitness = None
        self.selection_type = None
        self.final = False
        self.selected_in_generation = dict()

        self.constant_min = None
        self.constant_max = None
        self.random_min = None
        self.random_max = None

        self.initial_gp_depth_limit = None
        self.gp_terminal_node_generation_chance = None
        self.tournament_size = None
        self.min_tournament_size = None
        self.max_tournament_no_replacement_size = None
        self.max_tournament_size = None

        self.id = None

    def assign_id(self):
        # assigns a random id to self. Every unique GP Tree should call this once
        self.id = self.id = '{0}_{1}_{2}'.format('GPTree', str(id(self)), str(uuid.uuid4()))

    def proportional_selection(self, population, weights):
        # makes a random weighted selection from the population

        # raise an error if the lengths of the population and weights are different
        if len(population) != len(weights):
            raise IndexError

        # normalize the weights, if necessary
        normalized_weights = weights[:]
        min_weight = min(normalized_weights)
        if min_weight < 0:
            for i in range(len(normalized_weights)):
                normalized_weights[i] -= min_weight

        # build a list of the indices and cumulative selection weights
        indices_and_weights = []
        cum_weight = 0
        for i,w in enumerate(normalized_weights):
            cum_weight += w
            indices_and_weights.append((i, cum_weight))
        sum_weight = cum_weight

        # if the sum weight is 0 or inf, just return random candidate
        if sum_weight == 0 or sum_weight == math.inf or numpy.isnan(sum_weight):
            index = random.randrange(len(population))
            selection = population[index]
            return selection, index

        # select a number between 0 and the sum weight
        selection_number = random.uniform(0, sum_weight)

        # iterate through the items in the population until weights up to the selection number have passed, then return
        # the current item
        for i, w in indices_and_weights:
            if selection_number <= w:
                return population[i], i

        # if the program reaches this point, something is wrong. dump everything
        print('ERROR: EPPSEA tree overran proportional selection. Dumping objects in "errors"')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_directory = 'errors/error_{0}'.format(timestamp)
        os.makedirs(error_directory, exist_ok=True)

        with open(error_directory + '/population', 'wb') as file:
            pickle.dump(population, file)
        with open(error_directory + '/variables', 'w') as file:
            file.write('min_weight: {0}\n'.format(min_weight))
            file.write('weights: {0}\n'.format(weights))
            file.write('sum_weight: {0}\n'.format(sum_weight))
            file.write('selection_number: {0}\n'.format(selection_number))

    def tournament_selection(self, population, weights, k):
        # makes a k-tournament selection from the population
        if k > len(population) or k < 1:
            raise Exception('ERROR: trying to select from a population of {0} with a tournament size of {1}'.format(len(population), k))
        tournament_indices = random.sample(range(len(population)), k)
        index = max(tournament_indices, key=lambda i: weights[i])
        selection = population[index]
        return selection, index

    def truncation_selection(self, population, weights, n):
        # returns the population truncated to n members by the weights
        population_and_weights = zip(population, weights)
        population_and_weights = sorted(population_and_weights, key=lambda p: p[1], reverse=True)
        population_and_weights = population_and_weights[:n]
        return list(p[0] for p in population_and_weights)

    def stochastic_universal_sampling_selection(self, population, weights, n):
        # returns n members of the population selected using universal stochastic sampling

        # raise an error if the lengths of the population and weights are different
        if len(population) != len(weights):
            raise IndexError

        # normalize the weights, if necessary
        normalized_weights = weights[:]
        min_weight = min(normalized_weights)
        if min_weight < 0:
            for i in range(len(normalized_weights)):
                normalized_weights[i] -= min_weight

        # build a list of the indices and cumulative selection weights
        indices_and_weights = []
        cum_weight = 0
        for i,w in enumerate(normalized_weights):
            cum_weight += w
            indices_and_weights.append((i, cum_weight))
        sum_weight = cum_weight

        # if the sum weight is 0 or inf, just return random candidates
        if sum_weight == 0 or sum_weight == math.inf:
            selected = random.sample(population, n)
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

    def get_fitness_stats(self, fitnesses):
        # return [],0 if no fitnesses
        if len(fitnesses) == 0:
            return [], 0

        # determine the fitness rankings for each population member, and the sum fitness of the population (normalized, if negative)
        fitness_rankings = []
        for i in range(len(fitness_rankings)):
            fitness_rankings[i] += 1

        min_fitness = min(fitnesses)
        if min_fitness < 0:
            sum_fitness = sum(f - min_fitness for f in fitnesses)
        else:
            sum_fitness = sum(fitnesses)

        return fitness_rankings, sum_fitness

    def get_selectabilities(self, candidates, population_size, generation_number):
        # calculates the selectabilities of the candidates
        # returns a new list of tuples, each of (candidate, selectability)

        # sort the candidates by fitness (for calculating fitness rank)
        sorted_candidates = sorted(candidates, key=lambda c: c.fitness)

        # get fitness stats
        sum_fitness = sum(c.fitness for c in sorted_candidates)
        min_fitness = min(c.fitness for c in sorted_candidates)
        max_fitness = max(c.fitness for c in sorted_candidates)

        # calculate selectabilities
        terminal_values = dict()
        if 'fitness' in GPNode.terminals:
            terminal_values['fitness'] = numpy.array(list(c.fitness for c in sorted_candidates))
        if 'fitness_rank' in GPNode.terminals:
            terminal_values['fitness_rank'] = numpy.arange(1, len(sorted_candidates)+1)
        if 'sum_fitness' in GPNode.terminals:
            terminal_values['sum_fitness'] = numpy.repeat(sum_fitness, population_size)
        if 'min_fitness' in GPNode.terminals:
            terminal_values['min_fitness'] = numpy.repeat(min_fitness, population_size)
        if 'max_fitness' in GPNode.terminals:
            terminal_values['max_fitness'] = numpy.repeat(max_fitness, population_size)
        if 'relative_fitness' in GPNode.terminals:
            numpy.seterr(all='raise')
            if max_fitness == min_fitness or max_fitness == math.inf or min_fitness == math.inf:
                terminal_values['relative_fitness'] = numpy.repeat(1, population_size)
            else:
                terminal_values['relative_fitness'] = numpy.array(list(((c.fitness - min_fitness) / (max_fitness - min_fitness)) for c in sorted_candidates))
        if 'population_size' in GPNode.terminals:
            terminal_values['population_size'] = numpy.repeat(population_size, population_size)
        if 'birth_generation' in GPNode.terminals:
            terminal_values['birth_generation'] = numpy.array(list(c.birth_generation for c in sorted_candidates))

        if generation_number is not None:
            terminal_values['generation_number'] = numpy.repeat(generation_number, population_size)
        else:
            terminal_values['generation_number'] = numpy.repeat(0, population_size)

        if 'relative_uniqueness' in GPNode.terminals:
            all_genomes = numpy.stack(list(c.genome for c in sorted_candidates))
            average_genome = numpy.average(all_genomes, axis=0)
            distances_from_average_genome = numpy.array(list(scipy.spatial.distance.euclidean(g, average_genome) for g in all_genomes))
            max_distance = numpy.max(distances_from_average_genome)
            if max_distance == 0:
                max_distance = 1
            terminal_values['relative_uniqueness'] = distances_from_average_genome / max_distance

        selectabilities = self.get(terminal_values)

        # zip the candidates and selectabilities, and return
        return zip(sorted_candidates, selectabilities)

    def select(self, population, n=1, generation_number=None):
        # probabilistically selects n members of the population according to the selectability tree

        # raise an error if the population members do not have a fitness attribute
        if not all(hasattr(p, 'fitness') for p in population):
            raise Exception('EPPSEA ERROR: Trying to use an EEPSEA selector to select from a population'
                            'when one of the members does not have "fitness" defined.')

        # raise an error if the population members do not have a birth_generation attribute
        if 'birth_generation' in GPNode.terminals and not all(hasattr(p, 'birth_generation') for p in population):
            raise Exception('EPPSEA ERROR: Trying to use an EEPSEA selector to select from a population'
                            'when one of the members does not have "birth_generation" defined.')

        candidates = list(population)

        # prepare to catch an overflow error
        try:
            # get the candidates with selectabilities
            candidates_with_selectabilities = self.get_selectabilities(candidates, len(population), generation_number)
            # if EPPSEA overflows at any point, just return random choices

            # get the newly ordered lists of candidates and selectabilities
            candidates, selectabilities = zip(*candidates_with_selectabilities)
            candidates = list(candidates)
            selectabilities = list(selectabilities)

            # if we are doing stochastic universal sampling, select members all at once. Otherwise, select one at a time
            if self.selection_type == 'stochastic_universal_sampling':
                selected_members = self.stochastic_universal_sampling_selection(candidates, selectabilities, n)

            elif self.selection_type == 'truncation':
                selected_members = self.truncation_selection(candidates, selectabilities, n)

            else:
                selected_members = []
                for i in range(n):
                    if self.selection_type in ['proportional_replacement', 'proportional_no_replacement']:
                        selected_member, selected_index = self.proportional_selection(candidates, selectabilities)
                    elif self.selection_type in ['tournament_replacement', 'tournament_no_replacement']:
                        selected_member, selected_index = self.tournament_selection(candidates, selectabilities, self.tournament_size)
                    else:
                        raise Exception('EPPSEA ERROR: selection type {0} not found'.format(self.selection_type))

                    if self.selection_type in ['proportional_no_replacement','tournament_no_replacement']:
                        candidates.pop(selected_index)
                        selectabilities.pop(selected_index)

                    selected_members.append(selected_member)

            return selected_members

        except (OverflowError, FloatingPointError):
            return random.sample(candidates, n)

    def recombine(self, parent2):
        # recombines two GPTrees and returns a new child

        # copy the first parent
        new_child = copy.deepcopy(self)
        new_child.fitness = None
        new_child.assign_id()

        # select a point to insert a tree from the second parent
        insertion_point = random.choice(new_child.get_all_nodes())

        # copy a tree from the second parent
        replacement_tree = copy.deepcopy(random.choice(parent2.get_all_nodes()))

        # insert the tree
        new_child.replace_node(insertion_point, replacement_tree)
        
        # recombine misc options
        new_child.selection_type = random.choice([self.selection_type, parent2.selection_type])

        if self.tournament_size > parent2.tournament_size:
            new_child.tournament_size = random.randint(parent2.tournament_size, self.tournament_size)
        else:
            new_child.tournament_size = random.randint(self.tournament_size, parent2.tournament_size)

        # clamp the tournament size
            new_child.tournament_size = max(new_child.tournament_size, new_child.min_tournament_size)

        if new_child.selection_type in new_child.replacement_selections:
            new_child.tournament_size = min(new_child.tournament_size, new_child.max_tournament_size)
        else:
            new_child.tournament_size = min(new_child.tournament_size, new_child.max_tournament_no_replacement_size)

        return new_child

    def mutate(self):
        # replaces a randomly selected subtree with a new random subtree, and flips the misc options

        # select a point to insert a new random tree
        insertion_point = random.choice(self.get_all_nodes())

        # randomly generate a new subtree
        new_subtree = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
        new_subtree.grow(3, self.gp_terminal_node_generation_chance, None)

        # insert the new subtree
        self.replace_node(insertion_point, new_subtree)
        
        # chance to flip selection type
        if random.random() < 0.2:
            self.selection_type = random.choice(self.selection_types)

        self.tournament_size = round((self.tournament_size + random.randint(-5, 5)) * random.uniform(0.9, 1.1))

        # clamp the tournament size
        self.tournament_size = max(self.tournament_size, self.min_tournament_size)

        if self.selection_type in self.replacement_selections:
            self.tournament_size = min(self.tournament_size, self.max_tournament_size)
        else:
            self.tournament_size = min(self.tournament_size, self.max_tournament_no_replacement_size)

    def get(self, terminal_values):
        values = self.root.get(terminal_values)
        # if only a single value was returned, expand it into an array of repeated numbers
        if not(type(values) is numpy.ndarray):
            population_length = len(terminal_values['fitness'])
            values = numpy.repeat(values, population_length)
        return values
    
    def get_all_nodes(self):
        result = self.root.get_all_nodes()
        return result
    
    def get_all_nodes_depth_limited(self, depth_limit):
        result = self.root.get_all_nodes_depth_limited(depth_limit)
        return result

    def size(self):
        return len(self.get_all_nodes())
    
    def replace_node(self, node_to_replace, replacement_node):
        # replaces node in GPTree. Uses the replacement_node directly, not a copy of it
        if node_to_replace is self.root:
            self.root = replacement_node
            self.root.parent = None
        else:
            parent_of_replacement = node_to_replace.parent
            for i, c in enumerate(parent_of_replacement.children):
                if c is node_to_replace:
                    parent_of_replacement.children[i] = replacement_node
                    break
            replacement_node.parent = parent_of_replacement

    def simplify(self, target_node):
        # eliminates redundant branches in the tree

        # first, simplify the children
        if target_node.children is not None:
            for c in list(target_node.children):
                self.simplify(c)

        # go through the cases that can be simplified

        # anything multiplied by 1 is just itself
        if target_node.operation == '*':
            if target_node.children[0].operation == 'constant' and target_node.children[0].data == 1:
                new_node = copy.deepcopy(target_node.children[1])
                self.replace_node(target_node, new_node)
                return
            elif target_node.children[1].operation == 'constant' and target_node.children[1].data == 1:
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return

        # anything multiplied by 0 is 0
        if target_node.operation == '*':
            if target_node.children[0].operation == 'constant' and target_node.children[0].data == 0:
                target_node.operation = 'constant'
                target_node.data = 0
                self.children = None
            elif target_node.children[1].operation == 'constant' and target_node.children[1].data == 0:
                target_node.operation = 'constant'
                target_node.data = 0
                self.children = None

        # 0 divided by anything is 0
        if target_node.operation == '/':
            if target_node.children[0].operation == 'constant' and target_node.children[0].data == 0:
                target_node.operation = 'constant'
                target_node.data = 0
                self.children = None

        # anything divided by 1 is itself
        if target_node.operation == '/':
            if target_node.children[1].operation == 'constant' and target_node.children[1].data == 1:
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return

        # the absolute value of anything positive is always itself
        if target_node.operation == 'absolute':
            if target_node.children[0].operation == 'fitness_rank' or (target_node.children[0].operation == 'constant' and target_node.children[0].data >= 0):
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return

        # the maximum or minimum of two identical terminals is just that terminal
        if target_node.operation == 'max' or target_node.operation == 'min':
            if target_node.children[0].operation == target_node.children[1].operation and target_node.children[0].operation in GPNode.data_terminals:
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return

        # a step or division function on two identical terminators is always 1
        if target_node.operation == 'step' or target_node.operation == '/':
            if target_node.children[0].operation == target_node.children[1].operation and target_node.children[0].operation in GPNode.data_terminals:
                target_node.operation = 'constant'
                target_node.data = 1
                self.children = None

        # a subtraction function on two identical terminators is always 1
        if target_node.operation == '-':
            if target_node.children[0].operation == target_node.children[1].operation and target_node.children[0].operation in GPNode.data_terminals:
                target_node.operation = 'constant'
                target_node.data = 0
                self.children = None

        # the max, min, and step of constant terminals can be directly calculated
        if target_node.operation == 'max':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = max(target_node.children[0].data, target_node.children[1].data)
                self.children = None
        if target_node.operation == 'min':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = min(target_node.children[0].data, target_node.children[1].data)
                self.children = None
        if target_node.operation == 'step':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = int(target_node.children[0].data >= target_node.children[1].data)
                self.children = None

        # population_size will always be >= fitness_rank
        if target_node.operation == 'max':
            if target_node.children[0].operation == 'population_size' and target_node.children[1].operation == 'fitness_rank':
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return
            elif target_node.children[0].operation == 'fitness_rank' and target_node.children[1].operation == 'population_size':
                new_node = copy.deepcopy(target_node.children[1])
                self.replace_node(target_node, new_node)
                return
        if target_node.operation == 'min':
            if target_node.children[0].operation == 'population_size' and target_node.children[1].operation == 'fitness_rank':
                new_node = copy.deepcopy(target_node.children[1])
                self.replace_node(target_node, new_node)
                return
            elif target_node.children[0].operation == 'fitness_rank' and target_node.children[1].operation == 'population_size':
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return
        if target_node.operation == 'step':
            if target_node.children[0].operation == 'population_size' and target_node.children[1].operation == 'fitness_rank':
                target_node.operation = 'constant'
                target_node.data = 1
                self.children = None

        # max_fitness will always be >= min_fitness
        if target_node.operation == 'max':
            if target_node.children[0].operation == 'max_fitness' and target_node.children[1].operation == 'min_fitness':
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return
            elif target_node.children[0].operation == 'min_fitness' and target_node.children[1].operation == 'max_fitness':
                new_node = copy.deepcopy(target_node.children[1])
                self.replace_node(target_node, new_node)
                return
        if target_node.operation == 'min':
            if target_node.children[0].operation == 'max_fitness' and target_node.children[1].operation == 'min_fitness':
                new_node = copy.deepcopy(target_node.children[1])
                self.replace_node(target_node, new_node)
                return
            elif target_node.children[0].operation == 'min_fitness' and target_node.children[1].operation == 'max_fitness':
                new_node = copy.deepcopy(target_node.children[0])
                self.replace_node(target_node, new_node)
                return
        if target_node.operation == 'step':
            if target_node.children[0].operation == 'max_fitness' and target_node.children[1].operation == 'min_fitness':
                target_node.operation = 'constant'
                target_node.data = 1
                self.children = None

        # two constants can be added/subtracted/multiplied by each other
        if target_node.operation == '+':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = target_node.children[0].data + target_node.children[1].data
                self.children = None
        if target_node.operation == '-':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = target_node.children[0].data - target_node.children[1].data
                self.children = None
        if target_node.operation == '*':
            if target_node.children[0].operation == 'constant' and target_node.children[1].operation == 'constant':
                target_node.operation = 'constant'
                target_node.data = target_node.children[0].data * target_node.children[1].data
                self.children = None

    def randomize(self):
        if self.root is None:
            self.root = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
        self.root.grow(self.initial_gp_depth_limit, self.gp_terminal_node_generation_chance, None)

        self.selection_type = random.choice(self.selection_types)

        if self.selection_type in self.replacement_selections:
            self.tournament_size = random.randint(self.min_tournament_size, self.max_tournament_size)
        else:
            self.tournament_size = random.randint(self.min_tournament_size, self.max_tournament_no_replacement_size)

        self.assign_id()

    def verify_parents(self):
        for n in self.get_all_nodes():
            if n is self.root:
                assert(n.parent is None)
            else:
                assert(n.parent is not None)
                assert(n in n.parent.children)

    def get_string(self):
        result = self.root.get_string() + ' | selection type: {0} | '.format(self.selection_type)
        if self.selection_type in ['tournament_replacement', 'tournament_no_replacement']:
            result += ' | tournament size: {0}'.format(self.tournament_size)
        return result

    def get_code(self):
        return self.root.get_code()

    def get_dict(self):
        result = dict()
        result['select_with_replacement'] = self.select_with_replacement
        result['select_from_subset'] = self.select_from_subset
        result['selection_type'] = self.selection_type
        result['selection_subset_size'] = self.selection_subset_size
        result['constant_min'] = self.constant_min
        result['constant_max'] = self.constant_max
        result['random_min'] = self.random_min
        result['random_max'] = self.random_max
        result['id'] = self.id

        result['root'] = self.root.get_dict()

        return result

    def build_from_dict(self, d):
        self.fitness = None
        self.select_with_replacement = d['select_with_replacement']
        self.select_from_subset = d['select_from_subset']
        self.selection_type = d['selection_type']
        self.selection_subset_size = d['selection_subset_size']
        self.constant_min = d['constant_min']
        self.constant_max = d['constant_max']
        self.random_min = d['random_min']
        self.random_max = d['random_max']
        self.id = d['id']

        self.root = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
        self.root.build_from_dict(d['root'])

    def save_to_dict(self, filename):
        with open(filename, 'wb') as pickleFile:
            pickle.dump(self.get_dict(), pickleFile)

    def load_from_dict(self, filename):
        with open(filename, 'rb') as pickleFile:
            d = pickle.load(pickleFile)
            self.build_from_dict(d)


class EppseaSelectionFunction:
    # encapsulates all trees and functionality associated with one selection function
    def __init__(self, other=None):
        # if other is provided, copy variable values. Otherwise, initialize them all to none
        if other is not None:
            self.fitness = other.fitness
            self.mo_fitnesses = other.mo_fitnesses
            self.pareto_tier = other.pareto_tier
            self.gp_trees = other.gp_trees[:]

            self.number_of_selectors = other.number_of_selectors
            self.constant_min = other.constant_min
            self.constant_max = other.constant_max
            self.random_min = other.random_min
            self.random_max = other.random_max

            self.initial_gp_depth_limit = other.initial_gp_depth_limit
            self.gp_terminal_node_generation_chance = other.gp_terminal_node_generation_chance
            self.min_tournament_size = other.min_tournament_size
            self.max_tournament_no_replacement_size = other.max_tournament_no_replacement_size
            self.max_tournament_size = other.max_tournament_size


        else:
            self.fitness = None
            self.mo_fitnesses = None
            self.fitness_pre_parsimony = None
            self.mo_fitnesses_pre_parsimony = None
            self.pareto_tier = None
            self.gp_trees = None

            self.number_of_selectors = None
            self.constant_min = None
            self.constant_max = None
            self.random_min = None
            self.random_max = None

            self.initial_gp_depth_limit = None
            self.gp_terminal_node_generation_chance = None
            self.min_tournament_size = None
            self.max_tournament_no_replacement_size = None
            self.max_tournament_size = None

        # the id should never be copied, and should instead be reassigned with assign_id
        self.assign_id()

    def assign_id(self):
        # assigns a random id to self. Every unique EPPSEA individual should call this once
        self.id = '{0}_{1}_{2}'.format('EppseaSelectionFunction', str(id(self)), str(uuid.uuid4()))

    def randomize(self):
        # randomizes this individual and assigns a new id
        # clear the gp_trees
        self.gp_trees = []

        # create each new tree (one for parent selection, one for survival selection)
        for _ in range(self.number_of_selectors):
            new_gp_tree = GPTree()

            new_gp_tree.constant_min = self.constant_min
            new_gp_tree.constant_max = self.constant_max
            new_gp_tree.random_min = self.random_min
            new_gp_tree.random_max = self.random_max

            new_gp_tree.initial_gp_depth_limit = self.initial_gp_depth_limit
            new_gp_tree.gp_terminal_node_generation_chance = self.gp_terminal_node_generation_chance
            new_gp_tree.min_tournament_size = self.min_tournament_size
            new_gp_tree.max_tournament_no_replacement_size = self.max_tournament_no_replacement_size
            new_gp_tree.max_tournament_size = self.max_tournament_size

            new_gp_tree.randomize()

            self.gp_trees.append(new_gp_tree)

    def mutate(self):
        # mutates each gp tree
        for tree in self.gp_trees:
            tree.mutate()

    def recombine(self, parent2):
        # recombines each gp_tree belonging to this selection function
        # make a new selection function
        new_selection_function = EppseaSelectionFunction(self)

        # clear the list of gp_trees and recombine them
        new_selection_function.gp_trees = []
        for gp_tree1, gp_tree2 in zip(self.gp_trees, parent2.gp_trees):
            new_gptree = gp_tree1.recombine(gp_tree2)
            new_selection_function.gp_trees.append(new_gptree)

        # clear the fitness rating
        new_selection_function.fitness = None
        new_selection_function.mo_fitnesses = None
        return new_selection_function

    def select(self, population, n=1, selector=0, generation_number=None):
        return self.gp_trees[selector].select(population, n, generation_number)

    def simplify(self):
        for t in self.gp_trees:
            t.simplify(t.root)

    def get_string(self):
        tree_strings = list(tree.get_string() for tree in self.gp_trees)
        return ' |||| '.join(tree_strings)

    def gp_trees_size(self):
        # returns the size of all gp_trees
        size = 0
        for tree in self.gp_trees:
            size += tree.size()
        return size


class Eppsea:
    def __init__(self, config_path=None):

        # set up the results directory
        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_" + str(present_time)
        results_directory = "./results/eppsea/" + experiment_name
        os.makedirs(results_directory)
        self.results_directory = results_directory

        # setup logging file
        self.log_file_path = self.results_directory + '/log.txt'

        # try to read a config file from the config file path
        # if we do not have a config file, generate and use a default config
        if config_path is None:
            self.log('No config file path provided. Generating and using default config.', 'INFO')
            config_path = 'config/base_config/default.cfg'
            self.generate_default_config(config_path)
        # if the provided file path does not exist, generate and use a default config
        elif not os.path.isfile(str(config_path)):
            self.log('No config file found at {0}. Generating and using default config.'.format(config_path), 'INFO')
            config_path = 'config/base_config/default.cfg'
            self.generate_default_config(config_path)
        else:
            self.log('Using config file {0}'.format(config_path), 'INFO')

        # copy the used config file to the results path
        shutil.copyfile(config_path, results_directory + '/config_used.cfg')

        # set up the configuration object
        config = configparser.ConfigParser()
        config.read(config_path)

        # get the parameters from the configuration file
        self.gp_mu = config.getint('metaEA', 'metaEA mu')
        self.gp_lambda = config.getint('metaEA', 'metaEA lambda')
        self.max_gp_evals = config.getint('metaEA', 'metaEA maximum fitness evaluations')
        self.initial_gp_depth_limit = config.getint('metaEA', 'metaEA GP tree initialization depth limit')
        self.gp_terminal_node_generation_chance = config.getfloat('metaEA', 'metaEA GP tree terminal node generation chance')
        self.gp_k_tournament_k = config.getint('metaEA', 'metaEA k-tournament size')
        self.gp_survival_selection = config.get('metaEA', 'metaEA survival selection')
        self.gp_parsimony_pressure = config.getfloat('metaEA', 'parsimony pressure')
        self.use_relative_parsimony_pressure = config.getboolean('metaEA', 'use relative parsimony pressure')
        self.kill_bad_individuals = config.getboolean('metaEA', 'kill bad individuals')
        self.multiobjective = config.getboolean('metaEA', 'multiobjective')
        self.number_of_objectives = config.getint('metaEA', 'number of objectives')

        self.terminate_max_evals = config.getboolean('metaEA', 'terminate on maximum evals')
        self.terminate_no_avg_fitness_change = config.getboolean('metaEA', 'terminate on no improvement in average fitness')
        self.terminate_no_best_fitness_change = config.getboolean('metaEA', 'terminate on no improvement in best fitness')
        self.no_change_termination_generations = config.getint('metaEA', 'generations to termination for no improvement')

        self.restart_no_avg_fitness_change = config.getboolean('metaEA', 'restart on no improvement in average fitness')
        self.restart_no_best_fitness_change = config.getboolean('metaEA', 'restart on no improvement in best fitness')
        self.no_change_restart_generations = config.getint('metaEA', 'generations to restart for no improvement')

        self.gp_mutation_rate = config.getfloat('metaEA', 'metaEA mutation rate')
        self.force_mutation_of_clones = config.getboolean('metaEA', 'force mutation of clones')

        self.pickle_every_population = config.getboolean('experiment', 'pickle every population')
        self.pickle_final_population = config.getboolean('experiment', 'pickle final population')

        self.number_of_selectors = config.getint('evolved selection', 'number of selectors')

        self.min_tournament_size = config.getint('evolved selection', 'minimum tournament size')
        self.max_tournament_no_replacement_size = config.getint('evolved selection', 'maximum tournament (no replacement) size')
        self.max_tournament_size = config.getint('evolved selection', 'maximum tournament size')

        self.constant_min = config.getfloat('evolved selection', 'constant min')
        self.constant_max = config.getfloat('evolved selection', 'constant max')
        
        self.random_min = config.getfloat('evolved selection', 'random min')
        self.random_max = config.getfloat('evolved selection', 'random max')

        selection_types = config.get('evolved selection', 'selection type').strip().split(',')
        selection_types = list(s.strip() for s in selection_types)
        for s in list(GPTree.selection_types):
            if s not in selection_types:
                GPTree.selection_types.remove(s)

        selection_terminals = config.get('evolved selection', 'selection terminals').strip().split(',')
        selection_terminals = list(s.strip() for s in selection_terminals)
        for s in list(GPNode.data_terminals):
            if s not in selection_terminals:
                GPNode.data_terminals.remove(s)
        for s in list(GPNode.numeric_terminals):
            if s not in selection_terminals:
                GPNode.numeric_terminals.remove(s)

        GPNode.terminals = GPNode.data_terminals + GPNode.numeric_terminals

        # create a dictionary for the results
        self.results = dict()
        self.results['eval_counts'] = []
        self.results['average_fitness'] = []
        self.results['best_fitness'] = []

        # setup evolution data structures
        self.population = None
        self.new_population = None
        self.gen_number = None
        self.evolution_finished = None
        self.gens_since_avg_fitness_improvement = None
        self.gens_since_best_fitness_improvement = None
        self.highest_average_fitness = None
        self.highest_best_fitness = None
        self.gp_evals = None
        self.restarting = None
        self.final_best_member = None
        self.fitness_assignments = dict()

        self.start_time = None
        self.time_elapsed = None

        # make a list for initial population members
        self.initial_population = []

        # seed RNG and record seed
        seed = config.get('experiment', 'seed')
        try:
            seed = int(seed)
        except ValueError:
            seed = int(time.time())
        random.seed(seed)
        self.log('Using random seed {0}'.format(seed), 'INFO')

    def log(self, message, message_type):
        # Builds a log message out of a timestamp, the passed message, and a message type, then prints the message and
        # writes it in the log_file

        # Get the timestamp
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Put together the full message
        full_message = '{0}| {1}: {2}'.format(timestamp, message_type, message)

        # Print and log the message
        print(full_message)
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(full_message)
            log_file.write('\n')

    def randomize_population(self):
        # Fills the population with "mu" randomized individuals, including initial population members
        self.population = list(self.initial_population)

        while len(self.population) < self.gp_mu:
            # generate a new selection function
            new_selection_function = EppseaSelectionFunction()

            # set parameters for new selection function
            new_selection_function.number_of_selectors = self.number_of_selectors
            new_selection_function.constant_min = self.constant_min
            new_selection_function.constant_max = self.constant_max
            new_selection_function.random_min = self.random_min
            new_selection_function.random_max = self.random_max
            new_selection_function.initial_gp_depth_limit = self.initial_gp_depth_limit
            new_selection_function.gp_terminal_node_generation_chance = self.gp_terminal_node_generation_chance

            new_selection_function.min_tournament_size = self.min_tournament_size
            new_selection_function.max_tournament_no_replacement_size = self.max_tournament_no_replacement_size
            new_selection_function.max_tournament_size = self.max_tournament_size

            # randomize the selection function and add it to the population
            new_selection_function.randomize()
            self.population.append(new_selection_function)

        # force mutation of clones
        if self.force_mutation_of_clones:
            new_pop_strings = []
            for i, p in enumerate(self.population):
                p_string = p.get_string()
                while p_string in new_pop_strings:
                    p.mutate()
                    p_string = p.get_string()
                new_pop_strings.append(p)

        # simplify the children
        for p in self.population:
            p.simplify()

        # mark the entire population as new
        self.new_population = self.population

    def start_evolution(self):
        self.log('Starting evolution', 'INFO')
        # record start time
        self.start_time = time.time()

        # initialize the population
        self.randomize_population()

        # check population uniqueness
        self.check_gp_population_uniqueness(self.population, 0.75)

        # start evolution variables
        self.gp_evals = 0
        self.gen_number = 0
        self.evolution_finished = False
        self.gens_since_avg_fitness_improvement = 0
        self.gens_since_best_fitness_improvement = 0
        self.highest_average_fitness = -math.inf
        self.mo_highest_average_fitnesses = [-math.inf] * self.number_of_objectives
        self.highest_best_fitness = -math.inf
        self.mo_highest_best_fitnesses = [-math.inf] * self.number_of_objectives
        self.restarting = False

    def next_generation(self):
        # make sure all population members have been assigned fitness values
        if self.multiobjective and not all(p.mo_fitnesses is not None for p in self.population):
            self.log('Attempting to advance to next generation before assigning fitnesses to all members', 'ERROR')
            return
        if not self.multiobjective and not all(p.fitness is not None for p in self.population):
            self.log('Attempting to advance to next generation before assigning fitness to all members', 'ERROR')
            return


        # log and increment generation
        self.log('Finished generation {0}'.format(self.gen_number), 'INFO')
        self.gen_number += 1

        # increment evaluation counter
        self.gp_evals += len(self.new_population)

        # record fitness assignments
        for p in self.new_population:
            self.fitness_assignments[p.get_string()] = p.fitness

        # kill bad population members, if configured to
        # bad population members are those with a fitness less than the first quartile minus 3 * the interquartile range
        if self.kill_bad_individuals:
            if self.multiobjective:
                to_be_removed = []
                for i in range(self.number_of_objectives):
                    fitnesses = list(p.mo_fitnesses[i] for p in self.population)
                    threshold = numpy.percentile(fitnesses, 25) - 3 * (numpy.percentile(fitnesses, 75) - numpy.percentile(fitnesses, 25))
                    to_be_removed.extend(p for p in self.population if p.mo_fitnesses[i] < threshold)
                self.population = list(p for p in self.population if p not in to_be_removed)

            else:
                fitnesses = list(p.fitness for p in self.population)
                threshold = numpy.percentile(fitnesses, 25) - 3 * (numpy.percentile(fitnesses, 75) - numpy.percentile(fitnesses, 25))
                self.population = list(p for p in self.population if p.fitness >= threshold)

        # apply parsimony pressure to newly evaluated individuals
        for p in self.new_population:
            if self.multiobjective:
                p.mo_fitnesses_pre_parsimony = p.mo_fitnesses
                for i in range(self.number_of_objectives):
                    if self.use_relative_parsimony_pressure:
                        max_fitness = max(p.mo_fitnesses[i] for p in self.population)
                        min_fitness = min(p.mo_fitnesses[i] for p in self.population)
                        p.mo_fitnesses[i] -= (max_fitness - min_fitness) * self.gp_parsimony_pressure * p.gp_trees_size()
                    else:
                        p.mo_fitnesses[i] -= self.gp_parsimony_pressure * p.gp_trees_size()
            else:
                p.fitness_pre_parsimony = p.fitness
                if self.use_relative_parsimony_pressure:
                    max_fitness = max(p.fitness for p in self.population)
                    min_fitness = min(p.fitness for p in self.population)
                    p.fitness -= (max_fitness - min_fitness) * self.gp_parsimony_pressure * p.gp_trees_size()
                else:
                    p.fitness -= self.gp_parsimony_pressure * p.gp_trees_size()

        # Update results
        self.results['eval_counts'].append(self.gp_evals)
        if self.multiobjective:
            mo_average_fitnesses = list(statistics.mean(p.mo_fitnesses[i] for p in self.population) for i in range(self.number_of_objectives))
            mo_best_fitnesses = list(max(p.mo_fitnesses[i] for p in self.population) for i in range(self.number_of_objectives))
            self.results['average_fitness'].append(mo_average_fitnesses)
            self.results['best_fitness'].append(mo_best_fitnesses)
        else:
            average_fitness = statistics.mean(p.fitness for p in self.population)
            best_fitness = max(p.fitness for p in self.population)
            self.results['average_fitness'].append(average_fitness)
            self.results['best_fitness'].append(best_fitness)

        # pickle the population, if configured to
        if self.pickle_every_population:
            pickle_directory = self.results_directory + '/pickledPopulations'
            pickle_file_path = pickle_directory + '/gen{0}'.format(self.gen_number)
            os.makedirs(pickle_directory, exist_ok=True)
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(self.population, pickle_file)

        # check termination and restart conditions
        self.check_termination_and_restart_conditions()

        if not self.evolution_finished:

            # if we are restarting, regenerate the population
            if self.restarting:
                self.randomize_population()

                self.gens_since_avg_fitness_improvement = 0
                self.gens_since_best_fitness_improvement = 0
                self.highest_average_fitness = -math.inf
                self.mo_highest_average_fitnesses = [-math.inf]*self.number_of_objectives
                self.highest_best_fitness = -math.inf
                self.mo_highest_best_fitnesses = [-math.inf] * self.number_of_objectives
                self.restarting = False

            # otherwise, do survival selection and generate the next generation
            else:
                if self.multiobjective:
                    self.pareto_sort_population()

                # survival selection
                if self.gp_survival_selection == 'random':
                    self.population = random.sample(self.population, self.gp_mu)
                elif self.gp_survival_selection == 'truncation':
                    if self.multiobjective:
                        self.population.sort(key=lambda p: p.pareto_tier)
                        self.population = self.population[:self.gp_mu]
                    else:
                        self.population.sort(key=lambda p: p.fitness, reverse=True)
                        self.population = self.population[:self.gp_mu]

                # new child generation
                self.new_population = []
                # first, generate a number of children by asexual split and mutation
                while len(self.new_population) < self.gp_lambda * self.gp_mutation_rate:
                    # parent selection
                    if self.multiobjective:
                        parent = min(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.pareto_tier)
                    else:
                        parent = max(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.fitness)
                    # split and mutation
                    new_child = copy.deepcopy(parent)
                    new_child.mutate()
                    self.new_population.append(new_child)

                # sexual reproduction
                while len(self.new_population) < self.gp_lambda:
                    # parent selection (k tournament)
                    if self.multiobjective:
                        parent1 = min(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.pareto_tier)
                        parent2 = min(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.pareto_tier)
                    else:
                        parent1 = max(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.fitness)
                        parent2 = max(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.fitness)
                    # recombination/mutation
                    new_child = parent1.recombine(parent2)
                    if random.random() < self.gp_mutation_rate:
                        new_child.mutate()
                    self.new_population.append(new_child)

                # simplify the children
                for p in self.new_population:
                    p.simplify()

                # if configured to, force mutation of children who are clones, or have been seen before
                if self.force_mutation_of_clones:
                    new_pop_strings = []
                    for i, p in enumerate(self.new_population):
                        p_string = p.get_string()
                        while p_string in self.fitness_assignments.keys() or p_string in new_pop_strings:
                            p.mutate()
                            p_string = p.get_string()
                        new_pop_strings.append(p_string)

                # extend population with new members
                self.population.extend(self.new_population)

        else:
            # pickle the final population, if configured to
            if self.pickle_final_population:
                pickle_directory = self.results_directory + '/pickledPopulations'
                pickle_file_path = pickle_directory + '/final'
                os.makedirs(pickle_directory, exist_ok=True)
                with open(pickle_file_path, 'wb') as pickle_file:
                    pickle.dump(self.population, pickle_file)

            # write the results
            with open(self.results_directory + '/results.csv', 'w') as resultFile:

                result_writer = csv.writer(resultFile)

                result_writer.writerow(['evals', 'average fitness', 'best fitness'])
                result_writer.writerow(self.results['eval_counts'])
                result_writer.writerow(self.results['average_fitness'])
                result_writer.writerow(self.results['best_fitness'])

            # find the best population member(s), log its string, and expose it/them
            if self.multiobjective:
                self.pareto_sort_population()
                self.final_best_members = list(p for p in self.population if p.pareto_tier == 0)
                self.log('String form of best members:', 'INFO')
                for p in self.final_best_members:
                    self.log(p.get_string(), 'INFO')

            else:
                self.final_best_member = max(self.population, key=lambda p: p.fitness)
                final_best_member_string = self.final_best_member.get_string()
                self.log('String form of best Popi: {0}'.format(final_best_member_string), 'INFO')

            # log time elapsed
            self.time_elapsed = time.time() - self.start_time
            self.log('Time elapsed: {0}'.format(self.time_elapsed), 'INFO')

        return

    def check_gp_population_uniqueness(self, population, warning_threshold):
        population_strings = list(p.get_string() for p in population)
        unique_strings = set(population_strings)
        uniqueness = len(unique_strings) / len(population_strings)
        if uniqueness <= warning_threshold:
            self.log('GP population uniqueness is at {0}%. Consider increasing mutation rate.'.format(round(uniqueness*100)), 'WARNING')

    def check_termination_and_restart_conditions(self):
        # this function checks whether the population needs to restart or terminate
        # it logs the reason and sets the self.restarting or self.evolution_finished

        average_fitness_improved = False
        best_fitness_improved = False

        if self.multiobjective:
            # iterate through all objectives to check if the average or best fitness for any objective has improved
            for i in range(self.number_of_objectives):
                average_fitness = statistics.mean(p.mo_fitnesses[i] for p in self.population)
                best_fitness = max(p.mo_fitnesses[i] for p in self.population)

                if average_fitness > self.mo_highest_average_fitnesses[i]:
                    average_fitness_improved = True
                    self.mo_highest_average_fitnesses[i] = average_fitness
                if best_fitness > self.mo_highest_best_fitnesses[i]:
                    best_fitness_improved = True
                    self.mo_highest_best_fitnesses[i]= best_fitness

        else:
            # caluclate the average and best fitnesses to see if they have improved
            average_fitness = statistics.mean(p.fitness for p in self.population)
            best_fitness = max(p.fitness for p in self.population)

            if average_fitness > self.highest_average_fitness:
                self.highest_average_fitness = average_fitness
                average_fitness_improved = True
            if best_fitness > self.highest_best_fitness:
                self.highest_best_fitness = best_fitness
                best_fitness_improved = True

        # increment or reset the counters for generations since fitness improvement, and check if termination/restarting is necessary
        if average_fitness_improved:
            self.gens_since_avg_fitness_improvement = 0
        else:
            self.gens_since_avg_fitness_improvement += 1
            if self.terminate_no_avg_fitness_change and self.gens_since_avg_fitness_improvement >= self.no_change_termination_generations:
                self.evolution_finished = True
            elif self.restart_no_avg_fitness_change and self.gens_since_avg_fitness_improvement >= self.no_change_restart_generations:
                self.log('Restarting evolution due to no improvement in average fitness', 'INFO')
                self.restarting = True

        if best_fitness_improved:
            self.gens_since_best_fitness_improvement = 0
        else:
            self.gens_since_best_fitness_improvement += 1
            if self.terminate_no_best_fitness_change and self.gens_since_best_fitness_improvement >= self.no_change_termination_generations:
                self.log('Terminating evolution due to no improvement in best fitness', 'INFO')
                self.evolution_finished = True
            elif self.restart_no_best_fitness_change and self.gens_since_best_fitness_improvement >= self.no_change_restart_generations:
                self.log('Restarting evolution due to no improvement in best fitness', 'INFO')
                self.restarting = True

        if self.terminate_max_evals and self.gp_evals >= self.max_gp_evals:
            self.log('Terminating evolution due to max evaluations reached', 'INFO')
            self.evolution_finished = True

    def pareto_optimal(self, x, y):
        # returns true if x is pareto dominant of y, and false otherwise
        if all(xf >= yf for xf,yf in zip(x.mo_fitnesses, y.mo_fitnesses)) and any(xf > yf for xf,yf in zip(x.mo_fitnesses, y.mo_fitnesses)):
            return True
        else:
            return False

    def pareto_sort_population(self):
        # sorts the population into a pareto optimal heirarchy, and assigns a pareto rank to each population member
        # build the pareto heirarchy
        pareto_heirarchy = []
        for p in self.population:
            self.insert_into_pareto_heirarchy(p, pareto_heirarchy)
        # assign tier numbers of population members
        for i, tier in enumerate(pareto_heirarchy):
            for p in tier:
                p.pareto_tier = i

    def insert_into_pareto_heirarchy(self, x, pareto_heirarchy, tier_num=0):
        # inserts x into the pareto heirarchy at tier_num. Recursively inserts x into lower tiers if it is dominated, or moves dominated members to lower tiers
        # if x is being inserted at the bottom of the heirarchy, create a new tier for it
        if tier_num >= len(pareto_heirarchy):
            pareto_heirarchy.append([x])
        # otherwise, only insert x into the current tier if it is non-dominated, and move any members it dominates to a lower tier
        else:
            for p in list(pareto_heirarchy[tier_num]):
                if self.pareto_optimal(x, p):
                    pareto_heirarchy[tier_num].remove(p)
                    self.insert_into_pareto_heirarchy(p, pareto_heirarchy, tier_num+1)
                elif self.pareto_optimal(p, x):
                    self.insert_into_pareto_heirarchy(x, pareto_heirarchy, tier_num+1)
                    return
                # if this point is reached, x is pareto cooptimal with the whole tier, so insert it
            pareto_heirarchy[tier_num].append(x)

    def generate_default_config(self, file_path):
        # generates a default configuration file and writes it to file_path
        with open(file_path, 'w') as file:
            file.writelines([
                '[experiment]\n',
                'seed: time\n',
                'pickle every population: True\n',
                'pickle final population: True\n',
                '\n',
                '[metaEA]\n',
                'metaEA mu: 20\n',
                'metaEA lambda: 10\n',
                'metaEA maximum fitness evaluations: 200\n',
                'metaEA k-tournament size: 8\n',
                'metaEA survival selection: truncation\n',
                'metaEA GP tree initialization depth limit: 3\n',
                'metaEA mutation rate: 0.01\n',
                'force mutation of clones: True\n'
                'terminate on maximum evals: True\n',
                'terminate on no improvement in average fitness: False\n',
                'terminate on no improvement in best fitness: False\n',
                'generations to termination for no improvement: 25\n',
                'restart on no improvement in average fitness: False\n',
                'restart on no improvement in best fitness: False\n',
                'generations to restart for no improvement: 5\n',
                'parsimony pressure: 1\n',
                '\n',
                '[evolved selection]\n',
                'selection type: evolved\n',
                'select with replacement: evolved\n',
                'select from subset: evolved\n',
                'initial selection subset size: 10\n'
            ])
