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


class GPNode:
    numeric_terminals = ['constant', 'random']
    data_terminals = ['fitness', 'fitness_rank', 'population_size', 'sum_fitness', 'min_fitness', 'max_fitness', 'relative_fitness', 'birth_generation', 'generation_number']
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

    def grow(self, depth_limit, parent):
        if depth_limit == 0:
            self.operation = random.choice(GPNode.numeric_terminals + GPNode.data_terminals)
        else:
            self.operation = random.choice(GPNode.numeric_terminals + GPNode.data_terminals + GPNode.non_terminals)

        if self.operation == 'constant':
            self.data = random.uniform(self.constant_min, self.constant_max)
        if self.operation in GPNode.non_terminals:
            self.children = []
            for i in range(GPNode.child_count[self.operation]):
                new_child_node = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
                new_child_node.grow(depth_limit - 1, self)
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
            denom = self.children[1].get(terminal_values)
            if denom == 0:
                denom = 0.00001
            return self.children[0].get(terminal_values) / denom

        elif self.operation == 'step':
            if self.children[0].get(terminal_values) >= self.children[1].get(terminal_values):
                return 1
            else:
                return 0

        elif self.operation == 'absolute':
            return abs(self.children[0].get(terminal_values))

        elif self.operation == 'min':
            return min(self.children[0].get(terminal_values), self.children[1].get(terminal_values))

        elif self.operation == 'max':
            return max(self.children[0].get(terminal_values), self.children[1].get(terminal_values))

        elif self.operation in GPNode.data_terminals:
            return terminal_values[self.operation]

        elif self.operation == 'constant':
            return self.data

        elif self.operation == 'random':
            return random.uniform(self.random_min, self.random_max)

    def get_string(self):
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

    def get_code(self):
        # Returns executable python code for getting the selection chance from a population member p
        result = ''
        if self.operation in GPNode.non_terminals:
            if len(self.children) == 2:
                if self.operation in ('+', '-', '*'):
                    result = '(' + self.children[0].get_code() + self.operation + self.children[1].get_code() + ')'
                elif self.operation == '/':
                    result = '(' + self.children[0].get_code() + self.operation + '(0.000001+' + self.children[1].get_code() + '))'
                elif self.operation == 'step':
                    result = '(int(' + self.children[0].get_code() + '>=' + self.children[1].get_code() + '))'
                elif self.operation == 'absolute':
                    result = 'abs( '+ self.children[0].get_code() +' )'

        elif self.operation in GPNode.data_terminals:
            result = '(p.' + self.operation + ')'

        elif self.operation == 'constant':
            result = '(' + str(self.data) + ')'
        elif self.operation == 'random':
            result = '(random.random({0}, {1}))'.format(self.random_min, self.random_max)
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
    selection_types = ['proportional', 'maximum', 'stochastic_universal_sampling']

    def __init__(self):
        self.root = None
        self.fitness = None
        self.reusing_parents = None
        self.select_from_subset = None
        self.selection_type = None
        self.selection_subset_size = None
        self.final = False
        self.selected_in_generation = dict()

        self.constant_min = None
        self.constant_max = None
        self.random_min = None
        self.random_max = None

        self.initial_gp_depth_limit = None
        self.initial_selection_subset_size = None

        self.id = None

    def assign_id(self):
        # assigns a random id to self. Every unique GP Tree should call this once
        self.id = self.id = '{0}_{1}_{2}'.format('GPTree', str(id(self)), str(uuid.uuid4()))

    def proportional_selection(self, population, weights, subset_size):
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

        # determine the indeces of selectable candidates
        if subset_size is not None:
            selectable_indices = random.sample(range(len(population)), subset_size)
        else:
            selectable_indices = range(len(population))

        # build a list of the indices and cumulative selection weights
        indices_and_weights = []
        cum_weight = 0
        for i in selectable_indices:
            cum_weight += normalized_weights[i]
            indices_and_weights.append((i, cum_weight))
        sum_weight = cum_weight

        # if the sum weight is 0 or inf, just return random candidate
        if sum_weight == 0 or sum_weight == math.inf:
            selection = random.choice(population)
            index = population.index(selection)
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
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        error_directory = 'errors/error_{0}'.format(timestamp)
        os.makedirs(error_directory, exist_ok=True)

        with open(error_directory + '/population', 'wb') as file:
            pickle.dump(population, file)
        with open(error_directory + '/variables', 'w') as file:
            file.write('min_weight: {0}\n'.format(min_weight))
            file.write('weights: {0}\n'.format(weights))
            file.write('subset_size: {0}\n'.format(subset_size))
            file.write('selectable_indices: {0}\n'.format(selectable_indices))
            file.write('cum_weight: {0}\n'.format(cum_weight))
            file.write('sum_weight: {0}\n'.format(sum_weight))
            file.write('selection_number: {0}\n'.format(selection_number))

    def maximum_selection(self, population, weights, subset_size):
        # returns the member of the population for which the corresponding entry in weights is maximum

        # raise an error if the lengths of the population and weights are different
        if len(population) != len(weights):
            raise IndexError

        # determine the indeces of selectable candidates
        if subset_size is not None:
            selectable_indices = list(random.sample(range(len(population)), subset_size))
        else:
            selectable_indices = list(range(len(population)))

        # find the index of the weight with the maximum value
        index_of_max = max(selectable_indices, key=lambda i: weights[i])

        # return the population member at that index
        return population[index_of_max], index_of_max

    def stochastic_universal_sampling_selection(self, population, weights, subset_size, n):
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

        # determine the indeces of selectable candidates
        if subset_size is not None:
            selectable_indices = random.sample(range(len(population)), subset_size)
        else:
            selectable_indices = range(len(population))

        # build a list of the indices and cumulative selection weights
        indices_and_weights = []
        cum_weight = 0
        for i in selectable_indices:
            cum_weight += normalized_weights[i]
            indices_and_weights.append((i, cum_weight))
        sum_weight = cum_weight

        # if the sum weight is 0 or inf, just return random candidates
        if sum_weight == 0 or sum_weight == math.inf:
            selected = []
            for _ in range(n):
                selected_index = random.choice(selectable_indices)
                selected.append(population[selected_index])
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
        selectabilities = []
        for i in range(len(sorted_candidates)):
            terminal_values = dict()
            terminal_values['fitness'] = sorted_candidates[i].fitness
            terminal_values['fitness_rank'] = i+1
            terminal_values['sum_fitness'] = sum_fitness
            terminal_values['min_fitness'] = min_fitness
            terminal_values['max_fitness'] = max_fitness
            terminal_values['relative_fitness'] = (sorted_candidates[i].fitness - min_fitness) / (max_fitness - min_fitness)
            terminal_values['population_size'] = population_size
            terminal_values['birth_generation'] = sorted_candidates[i].birth_generation

            if generation_number is not None:
                terminal_values['generation_number'] = generation_number
            else:
                terminal_values['generation_number'] = 0

            selectabilities.append(self.get(terminal_values))

        # zip the candidates and selectabilities, and return
        return zip(sorted_candidates, selectabilities)

    def select(self, population, n=1, generation_number=None):
        # probabilistically selects n members of the population according to the selectability tree

        # raise an error if the population members do not have a fitness attribute
        if not all(hasattr(p, 'fitness') for p in population):
            raise Exception('EPPSEA ERROR: Trying to use an EEPSEA selector to select from a population'
                            'when one of the members does not have "fitness" defined.')

        # raise an error if the population members do not have a birth_generation attribute
        if not all(hasattr(p, 'birth_generation') for p in population):
            raise Exception('EPPSEA ERROR: Trying to use an EEPSEA selector to select from a population'
                            'when one of the members does not have "birth_generation" defined.')

        # create a list of selected individuals for the current generation if one does not exist
        if generation_number not in self.selected_in_generation.keys():
            self.selected_in_generation[generation_number] = list()

        # determine the list of candidates for selection
        if not self.reusing_parents and generation_number is not None:
            candidates = list(p for p in population if p not in self.selected_in_generation[generation_number])
        else:
            candidates = list(population)

        # prepare to catch an overflow error
        try:
            # get the candidates with selectabilities
            candidates_with_selectabilities = self.get_selectabilities(candidates, len(population), generation_number)

            # get the newly ordered lists of candidates and selectabilities
            candidates, selectabilities = zip(*candidates_with_selectabilities)
            candidates = list(candidates)
            selectabilities = list(selectabilities)

            # select, record, and return population members
            selected_members = []

            # if we are doing stochastic universal sampling, select members all at once. Otherwise, select one at a time
            if self.selection_type == 'stochastic_universal_sampling':

                if len(candidates) == 0:
                    raise Exception('EPPSEA ERROR: There are no candidates available for selection. '
                                    ' If mu < 2*lambda in your EA, make sure "select with replacement" is set to True'
                                    ' in your EPPSEA configuration, or handle this special case in your evaluation'
                                    ' of EPPSEA functions.')

                if self.select_from_subset and self.selection_subset_size < len(candidates):
                    subset_size = self.selection_subset_size
                else:
                    subset_size = None

                selected_members = self.stochastic_universal_sampling_selection(candidates, selectabilities, subset_size, n)
            else:
                for i in range(n):
                    if len(candidates) == 0:
                        raise Exception('EPPSEA ERROR: There are no candidates available for selection. '
                                        ' If mu < 2*lambda in your EA, make sure "select with replacement" is set to True'
                                        ' in your EPPSEA configuration, or handle this special case in your evaluation'
                                        ' of EPPSEA functions.')

                    if self.select_from_subset and self.selection_subset_size < len(candidates):
                        subset_size = self.selection_subset_size
                    else:
                        subset_size = None

                    if self.selection_type == 'proportional':
                        selected_member, selected_index = self.proportional_selection(candidates, selectabilities, subset_size)
                    elif self.selection_type == 'maximum':
                        selected_member, selected_index = self.maximum_selection(candidates, selectabilities, subset_size)
                    else:
                        raise Exception('EPPSEA ERROR: selection type {0} not found'.format(self.selection_type))

                    if generation_number is not None:
                        self.selected_in_generation[generation_number].append(selected_member)

                    if not self.reusing_parents:
                        candidates.pop(selected_index)
                        selectabilities.pop(selected_index)

                    selected_members.append(selected_member)

            return selected_members

        # if EPPSEA overflows at any point, just return random choices
        except OverflowError:
            print('WARNING: EPPSEA Overflow. Returning random selection')
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
        new_child.reusing_parents = random.choice([self.reusing_parents, parent2.reusing_parents])
        new_child.select_from_subset = random.choice(([self.select_from_subset, parent2.select_from_subset]))

        if self.selection_subset_size > parent2.selection_subset_size:
            new_child.selection_subset_size = random.randint(parent2.selection_subset_size, self.selection_subset_size)
        else:
            new_child.selection_subset_size = random.randint(self.selection_subset_size, parent2.selection_subset_size)

        return new_child

    def mutate(self):
        # replaces a randomly selected subtree with a new random subtree, and flips the misc options

        # select a point to insert a new random tree
        insertion_point = random.choice(self.get_all_nodes())

        # randomly generate a new subtree
        new_subtree = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
        new_subtree.grow(3, None)

        # insert the new subtree
        self.replace_node(insertion_point, new_subtree)
        
        # 50/50 chance to flip misc options
        if random.random() < 0.5:
            self.selection_type = random.choice(self.selection_types)
        if random.random() < 0.5:
            self.reusing_parents = not self.reusing_parents
        if random.random() < 0.5:
            self.select_from_subset = not self.select_from_subset

        self.selection_subset_size = max(1, round((self.selection_subset_size + random.randint(-5, 5)) * random.uniform(0.9, 1.1)))
        
    def get(self, terminal_values):
        return self.root.get(terminal_values)
    
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

    def randomize(self):
        if self.root is None:
            self.root = GPNode(self.constant_min, self.constant_max, self.random_min, self.random_max)
        self.root.grow(self.initial_gp_depth_limit, None)

        self.selection_type = random.choice(self.selection_types)
        self.reusing_parents = bool(random.random() < 0.5)
        self.select_from_subset = bool(random.random() < 0.5)
        self.selection_subset_size = self.initial_selection_subset_size

        self.assign_id()

    def verify_parents(self):
        for n in self.get_all_nodes():
            if n is self.root:
                assert(n.parent is None)
            else:
                assert(n.parent is not None)
                assert(n in n.parent.children)

    def get_string(self):
        return self.root.get_string() + ' | selection type: {0} | reusing parents: {1} | select from subset: {2} | selection_subset_size: {3}'.format(self.selection_type, self.reusing_parents, self.select_from_subset, self.selection_subset_size)

    def get_code(self):
        return self.root.get_code()

    def get_dict(self):
        result = dict()
        result['reusing_parents'] = self.reusing_parents
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
        self.reusing_parents = d['reusing_parents']
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

    def is_clone(self, population):
        # returns true if this GPTree is a clone of any members of the given population
        # uses the get_string() function of the GPTree, so there may be some false negatives, but no false positives
        for p in population:
            if self is not p and self.get_string() == p.get_string():
                return True
        return False


class EppseaSelectionFunction:
    # encapsulates all trees and functionality associated with one selection function
    def __init__(self, other=None):
        # if other is provided, copy variable values. Otherwise, initialize them all to none
        if other is not None:
            self.fitness = other.fitness
            self.gp_trees = other.gp_trees[:]


            self.constant_min = other.constant_min
            self.constant_max = other.constant_max
            self.random_min = other.random_min
            self.random_max = other.random_max

            self.initial_gp_depth_limit = other.initial_gp_depth_limit
            self.initial_selection_subset_size = other.initial_selection_subset_size
        else:
            self.fitness = None
            self.gp_trees = None

            self.constant_min = None
            self.constant_max = None
            self.random_min = None
            self.random_max = None

            self.initial_gp_depth_limit = None
            self.initial_selection_subset_size = None

        # the id should never be copied, and should instead be reassigned with assign_id
        self.id = None

    def assign_id(self):
        # assigns a random id to self. Every unique EPPSEA individual should call this once
        self.id = '{0}_{1}_{2}'.format('EppseaSelectionFunction', str(id(self)), str(uuid.uuid4()))

    def randomize(self):
        # randomizes this individual and assigns a new id
        # clear the gp_trees
        self.gp_trees = []

        # create each new tree (only one for now)
        new_gp_tree = GPTree()

        new_gp_tree.constant_min = self.constant_min
        new_gp_tree.constant_max = self.constant_max
        new_gp_tree.random_min = self.random_min
        new_gp_tree.random_max = self.random_max

        new_gp_tree.initial_gp_depth_limit = self.initial_gp_depth_limit
        new_gp_tree.initial_selection_subset_size = self.initial_selection_subset_size

        new_gp_tree.randomize()
        self.gp_trees.append(new_gp_tree)

    def mutate(self):
        # mutates each gp tree
        for tree in self.gp_trees:
            tree.mutate()

    def recombine(self, parent2):
        # recombines each gp_tree belonging to this selection function
        # make a new selection function
        new_selection_function = EppseaSelectionFunction()

        # clear the list of gp_trees and recombine them
        new_selection_function.gp_trees = []
        for gp_tree1, gp_tree2 in zip(self.gp_trees, parent2.gp_trees):
            new_gptree = gp_tree1.recombine(gp_tree2)
            new_selection_function.gp_trees.append(new_gptree)

        # clear the fitness rating and assign a new id
        new_selection_function.assign_id()
        new_selection_function.fitness = None
        return new_selection_function

    def select(self, population, n=1, generation_number=None):
        return self.gp_trees[0].select(population, n, generation_number)

    def is_clone(self, population):
        # returns true if this selection function is a clone of any other selection function in the population
        trees = (p.gp_trees[0] for p in population)
        return self.gp_trees[0].is_clone(trees)

    def get_string(self):
        # for now, just gets the string of the only gp_tree
        return self.gp_trees[0].get_string()

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
        self.gp_k_tournament_k = config.getint('metaEA', 'metaEA k-tournament size')
        self.gp_survival_selection = config.get('metaEA', 'metaEA survival selection')
        self.gp_parsimony_pressure = config.getfloat('metaEA', 'parsimony pressure')

        self.terminate_max_evals = config.getboolean('metaEA', 'terminate on maximum evals')
        self.terminate_no_avg_fitness_change = config.getboolean('metaEA', 'terminate on no improvement in average fitness')
        self.terminate_no_best_fitness_change = config.getboolean('metaEA', 'terminate on no improvement in best fitness')
        self.no_change_termination_generations = config.get('metaEA', 'generations to termination for no improvement')

        self.restart_no_avg_fitness_change = config.getboolean('metaEA', 'restart on no improvement in average fitness')
        self.restart_no_best_fitness_change = config.getboolean('metaEA', 'restart on no improvement in best fitness')
        self.no_change_restart_generations = config.getint('metaEA', 'generations to restart for no improvement')

        self.gp_mutation_rate = config.getfloat('metaEA', 'metaEA mutation rate')
        self.force_mutation_of_clones = config.getboolean('metaEA', 'force mutation of clones')

        self.pickle_every_population = config.getboolean('experiment', 'pickle every population')
        self.pickle_final_population = config.getboolean('experiment', 'pickle final population')

        try:
            self.force_reusing_parents = config.getboolean('evolved selection', 'select with replacement')
        except ValueError:
            self.force_reusing_parents = None

        try:
            self.force_select_from_subset = config.getboolean('evolved selection', 'select from subset')
        except ValueError:
            self.force_select_from_subset = None

        self.initial_selection_subset_size = config.getint('evolved selection', 'initial selection subset size')

        self.constant_min = config.getfloat('evolved selection', 'constant min')
        self.constant_max = config.getfloat('evolved selection', 'constant max')
        
        self.random_min = config.getfloat('evolved selection', 'random min')
        self.random_max = config.getfloat('evolved selection', 'random max')

        selection_type = config.get('evolved selection', 'selection type')
        if selection_type not in GPTree.selection_types and selection_type != 'evolved':
            self.log('Evolved selection type {0} not found in available selection types. Use one of {1} or "evolved" for selection type'.format(selection_type, str(GPTree.selection_types)), 'ERROR')
            raise Exception('EPPSEA ERROR: See log file')
        elif selection_type != 'evolved':
            self.force_selection_type = selection_type
        else:
            self.force_selection_type = None

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

        self.start_time = None
        self.time_elapsed = None

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
        # Fills the population with "mu" randomized individuals
        self.population = []

        for i in range(self.gp_mu):
            # generate a new selection function
            new_selection_function = EppseaSelectionFunction()

            # set parameters for new selection function
            new_selection_function.constant_min = self.constant_min
            new_selection_function.constant_max = self.constant_max
            new_selection_function.random_min = self.random_min
            new_selection_function.random_max = self.random_max
            new_selection_function.initial_gp_depth_limit = self.initial_gp_depth_limit
            new_selection_function.initial_selection_subset_size = self.initial_selection_subset_size

            # randomize the selection function and add it to the population
            new_selection_function.randomize()
            self.population.append(new_selection_function)

            # force selection function settings, if configured to
            if self.force_selection_type is not None:
                new_selection_function.gp_trees[0].selection_type = self.force_selection_type
            if self.force_reusing_parents is not None:
                new_selection_function.gp_trees[0].reusing_parents = self.force_reusing_parents
            if self.force_select_from_subset is not None:
                new_selection_function.gp_trees[0].select_from_subset = self.force_select_from_subset

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
        self.highest_best_fitness = -math.inf
        self.restarting = False

    def next_generation(self):
        # make sure all population members have been assigned fitness values
        if not all(p.fitness is not None for p in self.population):
            self.log('Attempting to advance to next generation before assigning fitness to all members', 'ERROR')
            return

        # log and increment generation
        self.log('Finished generation {0}'.format(self.gen_number), 'INFO')
        self.gen_number += 1

        # increment evaluation counter
        self.gp_evals += len(self.new_population)

        # apply parsimony pressure
        for p in self.new_population:
            p.fitness -= self.gp_parsimony_pressure * p.gp_trees_size()

        # Update results
        average_fitness = statistics.mean(p.fitness for p in self.population)
        best_fitness = max(p.fitness for p in self.population)

        self.results['eval_counts'].append(self.gp_evals)
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
        if average_fitness > self.highest_average_fitness:
            self.highest_average_fitness = average_fitness
            self.gens_since_avg_fitness_improvement = 0
        else:
            self.gens_since_avg_fitness_improvement += 1
            if self.terminate_no_avg_fitness_change and self.gens_since_avg_fitness_improvement >= self.no_change_termination_generations:
                self.log('Terminating evolution due to no improvement in average fitness', 'INFO')
                self.evolution_finished = True
            elif self.restart_no_avg_fitness_change and self.gens_since_avg_fitness_improvement >= self.no_change_restart_generations:
                self.log('Restarting evolution due to no improvement in average fitness', 'INFO')
                self.restarting = True
        if best_fitness > self.highest_best_fitness:
            self.highest_best_fitness = best_fitness
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

        if not self.evolution_finished:

            # if we are restarting, regenerate the population
            if self.restarting:
                self.randomize_population()

                self.gens_since_avg_fitness_improvement = 0
                self.gens_since_best_fitness_improvement = 0
                self.highest_average_fitness = -math.inf
                self.highest_best_fitness = -math.inf
                self.restarting = False

            # otherwise, do survival selection and generate the next generation
            else:
                # survival selection
                if self.gp_survival_selection == 'random':
                    self.population = random.sample(self.population, self.gp_mu)
                elif self.gp_survival_selection == 'truncation':
                    self.population.sort(key=lambda p: p.fitness, reverse=True)
                    self.population = self.population[:self.gp_mu]

                # parent selection and new child generation
                self.new_population = []
                while len(self.new_population) < self.gp_lambda:
                    # parent selection (k tournament)
                    parent1 = max(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.fitness)
                    parent2 = max(random.sample(self.population, self.gp_k_tournament_k), key=lambda p: p.fitness)
                    # recombination/mutation
                    new_child = parent1.recombine(parent2)
                    if random.random() < self.gp_mutation_rate or (
                            self.force_mutation_of_clones and new_child.is_clone(self.population + self.new_population)):
                        new_child.mutate()
                    self.new_population.append(new_child)

                # extend population with new members
                self.population.extend(self.new_population)

            # force selection function settings, if configured to
            for p in self.population:
                if self.force_selection_type is not None:
                    p.selection_type = self.force_selection_type
                if self.force_reusing_parents is not None:
                    p.reusing_parents = self.force_reusing_parents
                if self.force_select_from_subset is not None:
                    p.select_from_subset = self.force_select_from_subset

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

            # find the best population member, log its string, and expose it
            self.final_best_member = max(self.population, key=lambda p: p.fitness)
            final_best_member_string = self.final_best_member.get_string()
            self.log('String form of best Popi: {0}'.format(final_best_member_string), 'INFO')

            # log time elapsed
            self.time_elapsed = time.time() - self.start_time
            self.log('Time elapsed: {0}'.format(self.time_elapsed), 'INFO')

        # force selection function settings, if configured to
        for p in self.new_population:
            if self.force_selection_type is not None:
                p.gp_trees[0].selection_type = self.force_selection_type
            if self.force_reusing_parents is not None:
                p.gp_trees[0].reusing_parents = self.force_reusing_parents
            if self.force_select_from_subset is not None:
                p.gp_trees[0].select_from_subset = self.force_select_from_subset

        return

    def check_gp_population_uniqueness(self, population, warning_threshold):
        population_strings = list(p.get_string() for p in population)
        unique_strings = set(population_strings)
        uniqueness = len(unique_strings) / len(population_strings)
        if uniqueness <= warning_threshold:
            self.log('GP population uniqueness is at {0}%. Consider increasing mutation rate.'.format(round(uniqueness*100)), 'WARNING')

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
