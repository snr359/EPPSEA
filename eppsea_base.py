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


class GPNode:
    numeric_terminals = ['constant'] #TODO: include random later?
    data_terminals = ['fitness', 'fitnessRank', 'populationSize', 'sumFitness']
    non_terminals = ['+', '-', '*', '/', 'step']
    child_count = {'+': 2, '-': 2, '*': 2, '/': 2, 'step': 2}

    def __init__(self):
        self.operation = None
        self.data = None
        self.children = None
        self.parent = None

    def grow(self, depth_limit, parent):
        if depth_limit == 0:
            self.operation = random.choice(GPNode.numeric_terminals + GPNode.data_terminals)
        else:
            self.operation = random.choice(GPNode.numeric_terminals + GPNode.data_terminals + GPNode.non_terminals)

        if self.operation == 'constant':
            self.data = random.expovariate(0.07)
        if self.operation in GPNode.non_terminals:
            self.children = []
            for i in range(GPNode.child_count[self.operation]):
                new_child_node = GPNode()
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

        elif self.operation in GPNode.data_terminals:
            return terminal_values[self.operation]

        elif self.operation == 'constant':
            return self.data

        elif self.operation == 'random':
            return random.expovariate(0.07)

    def get_string(self):
        if self.operation in GPNode.non_terminals:
            result = "(" + self.children[0].get_string() + " " + self.operation + " " + self.children[1].get_string() + ")"
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

        elif self.operation in GPNode.data_terminals:
            result = '(p.' + self.operation + ')'

        elif self.operation == 'constant':
            result = '(' + str(self.data) + ')'
        elif self.operation == 'random':
            result = '(random.expovariate(0.07))'
        return result

    def get_all_nodes(self):
        nodes = []
        nodes.append(self)
        if self.children is not None:
            for c in self.children:
                nodes.extend(c.get_all_nodes())
        return nodes

    def get_all_nodes_depth_limited(self, depthLimit):
        # returns all nodes down to a certain depth limit
        nodes = []
        nodes.append(self)
        if self.children is not None and depthLimit > 0:
            for c in self.children:
                nodes.extend(c.get_all_nodes_depth_limited(depthLimit - 1))
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
                new_node = GPNode()
                new_node.build_from_dict(c)
                new_node.parent = self
                self.children.append(new_node)
        else:
            self.children = None

class GPTree:
    # encapsulates a tree made of GPNodes that determine probability of selection, as well as other options relating
    # to parent selection
    selection_types = ['proportional', 'maximum']

    def __init__(self):
        self.root = None
        self.fitness = None
        self.reusing_parents = None
        self.select_from_subset = None
        self.selection_type = None
        self.selection_subset_size = None
        self.final = False
        self.selected_in_generation = dict()

    def proportional_selection(self, population, weights, subset_size):
        # makes a random weighted selection from the population

        # raise an error if the lengths of the population and weights are different
        if len(population) != len(weights):
            raise IndexError

        # normalize the weights, if necessary
        min_weight = min(weights)
        if min_weight < 0:
            for i in range(len(weights)):
                weights[i] -= min_weight

        # determine the indeces of selectable candidates
        if subset_size is not None:
            selectable_indices = list(random.sample(range(len(population)), subset_size))
        else:
            selectable_indices = list(range(len(population)))

        # calculate the sum weight and select a number between 0 and the sum weight
        sum_weight = sum(weights[i] for i in selectable_indices)
        selection_number = random.uniform(0, sum_weight)

        # iterate through the items in the population until weights up to the selection number have passed, then return
        # the current item
        for i in selectable_indices:
            if selection_number <= weights[i]:
                return population[i], i
            else:
                selection_number -= weights[i]


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

    def get_fitness_stats(self, fitnesses):
        # return [],0 if no fitnesses
        if len(fitnesses) == 0:
            return [], 0

        # determine the fitness rankings for each population member, and the sum fitness of the population (normalized, if negative)
        fitness_rankings = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x])
        for i in range(len(fitness_rankings)):
            fitness_rankings[i] += 1

        min_fitness = min(fitnesses)
        if min_fitness < 0:
            sum_fitness = sum(f - min_fitness for f in fitnesses)
        else:
            sum_fitness = sum(fitnesses)

        return fitness_rankings, sum_fitness

    def select(self, population, n=1, generation_num=None):
        # probabilistically selects n members of the population according to the selectability tree

        # raise an error if the population members do not have a fitness attribute
        if not all(hasattr(p, 'fitness') for p in population):
            raise Exception('EPPSEA ERROR: Trying to use an EEPSEA selector to select from a population'
                            'when one of the members does not have "fitness" defined.')

        # create a list of selected individuals for the current generation if one does not exist
        if generation_num is not None and generation_num not in self.selected_in_generation.keys():
            self.selected_in_generation[generation_num] = list()

        # determine the list of candidates for selection
        if not self.reusing_parents and generation_num is not None:
            candidates = list(p for p in population if p not in self.selected_in_generation[generation_num])
        else:
            candidates = list(population)

        # get the fitnesses from the candidates
        fitnesses = list(p.fitness for p in candidates)

        # get the fitness rankings and the (normalized) sum of fitnesses
        fitness_rankings, sum_fitness = self.get_fitness_stats(tuple(fitnesses))

        # determine the selectability for each population member
        selectabilities = []
        for i in range(len(candidates)):
            terminal_values = dict()
            terminal_values['fitness'] = fitnesses[i]
            terminal_values['fitnessRank'] = fitness_rankings[i]
            terminal_values['sumFitness'] = sum_fitness
            terminal_values['populationSize'] = len(population)
            selectabilities.append(self.get(terminal_values))

        # select, record, and return population members
        selected_members = []
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

            if generation_num is not None:
                self.selected_in_generation[generation_num].append(selected_member)

            if not self.reusing_parents:
                candidates.pop(selected_index)
                selectabilities.pop(selected_index)

            selected_members.append(selected_member)

        return selected_members

    def recombine(self, parent2):
        # recombines two GPTrees and returns a new child

        # copy the first parent
        new_child = copy.deepcopy(self)
        new_child.fitness = None

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
        new_subtree = GPNode()
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

        self.selection_subset_size = round((self.selection_subset_size + random.randint(-5,5)) * random.uniform(0.9, 1.1))
        
    def get(self, terminal_values):
        return self.root.get(terminal_values)
    
    def get_all_nodes(self):
        result = self.root.get_all_nodes()
        return result
    
    def get_all_nodes_depth_limited(self, depth_limit):
        result = self.root.get_all_nodes_depth_limited(depth_limit)
        return result
    
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

    def randomize(self, initial_depth_limit, initial_selection_subset_size):
        if self.root is None:
            self.root = GPNode()
        self.root.grow(initial_depth_limit, None)

        self.selection_type = random.choice(self.selection_types)
        self.reusing_parents = bool(random.random() < 0.5)
        self.select_from_subset = bool(random.random() < 0.5)
        self.selection_subset_size = initial_selection_subset_size

    def verify_parents(self):
        for n in self.get_all_nodes():
            if n is self.root:
                assert(n.parent is None)
            else:
                assert(n.parent is not None)
                assert(n in n.parent.children)

    def get_string(self):
        return self.root.get_string() + ' | selection type: {0} | reusing parents: {1} | select from subset: {2} | selection_subset_size: {3}'.format(self.selection_type, self.reusing_parents, self.select_from_subset, self.selection_subset_size)

    def getCode(self):
        return self.root.get_code()

    def get_dict(self):
        return self.root.get_dict()

    def build_from_dict(self, d):
        self.fitness = None
        self.root = GPNode()
        self.root.build_from_dict(d)

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

class Eppsea:
    def __init__(self, config_path=None):

        # set up the results directory
        present_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = "eppsea_" + str(present_time)
        results_directory = "./results/eppsea/" + experiment_name
        os.makedirs(results_directory)
        self.results_directory = results_directory

        # setup logging file
        self.log_file = open(self.results_directory + '/log.txt', 'w')

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

    # record start time
    start_time = time.time()

    def log(self, message, message_type):
        # Builds a log message out of a timestamp, the passed message, and a message type, then prints the message and
        # writes it in the log_file

        # Get the timestamp
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Put together the full message
        full_message = '{0}| {1}: {2}'.format(timestamp, message_type, message)

        # Print and log the message
        print(full_message)
        self.log_file.write(full_message)
        self.log_file.write('\n')

    def start_evolution(self):
        self.log('Starting evolution', 'INFO')
        # record start time
        self.start_time = time.time()

        # initialize the population
        self.population = []
        for i in range(self.gp_mu):
            new_tree = GPTree()
            new_tree.randomize(self.initial_gp_depth_limit, self.initial_selection_subset_size)
            self.population.append(new_tree)

            # force selection function settings, if configured to
            if self.force_selection_type is not None:
                new_tree.selection_type = self.force_selection_type
            if self.force_reusing_parents is not None:
                new_tree.reusing_parents = self.force_reusing_parents
            if self.force_select_from_subset is not None:
                new_tree.select_from_subset = self.force_select_from_subset

        # mark the entire population as new
        self.new_population = list(self.population)

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
                self.population = []
                for i in range(self.gp_mu):
                    new_tree = GPTree()
                    new_tree.randomize(self.initial_gp_depth_limit, self.initial_selection_subset_size)
                    self.population.append(new_tree)

                    # force selection function settings, if configured to
                    if self.force_selection_type is not None:
                        new_tree.selection_type = self.force_selection_type
                    if self.force_reusing_parents is not None:
                        new_tree.reusing_parents = self.force_reusing_parents
                    if self.force_select_from_subset is not None:
                        new_tree.select_from_subset = self.force_select_from_subset

                self.new_population = list(self.population)

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

        return

    def check_gp_population_uniqueness(self, population, warningThreshold):
        population_strings = list(p.get_string() for p in population)
        unique_strings = set(population_strings)
        uniqueness = len(unique_strings) / len(population_strings)
        if uniqueness <= warningThreshold:
            self.log('GP population uniqueness is at {0}%. Consider increasing mutation rate.'.format(round(uniqueness*100)), 'WARNING')

    def generate_default_config(self, filePath):
        # generates a default configuration file and writes it to filePath
        with open(filePath, 'w') as file:
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
                '\n',
                '[evolved selection]\n',
                'selection type: evolved\n',
                'select with replacement: evolved\n',
                'select from subset: evolved\n',
                'initial selection subset size: 10\n'
            ])
