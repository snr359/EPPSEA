import sys
import time
import copy
import random
import datetime
import os
import csv
import math
import multiprocessing
import configparser
import shutil
import pickle
import statistics


class GPNode:
    numericTerminals = ['constant'] #TODO: include random later?
    dataTerminals = ['fitness', 'fitnessRank', 'populationSize', 'sumFitness']
    nonTerminals = ['+', '-', '*', '/', 'step']
    childCount = {'+': 2, '-': 2, '*': 2, '/': 2, 'step': 2}

    def __init__(self):
        self.operation = None
        self.data = None
        self.children = None
        self.parent = None

    def grow(self, depthLimit, parent):
        if depthLimit == 0:
            self.operation = random.choice(GPNode.numericTerminals + GPNode.dataTerminals)
        else:
            self.operation = random.choice(GPNode.numericTerminals + GPNode.dataTerminals + GPNode.nonTerminals)

        if self.operation == 'constant':
            self.data = random.expovariate(0.07)
        if self.operation in GPNode.nonTerminals:
            self.children = []
            for i in range(GPNode.childCount[self.operation]):
                newChildNode = GPNode()
                newChildNode.grow(depthLimit-1, self)
                self.children.append(newChildNode)
        self.parent = parent

    def get(self, terminalValues):
        if self.operation == '+':
            return self.children[0].get(terminalValues) + self.children[1].get(terminalValues)
        elif self.operation == '-':
            return self.children[0].get(terminalValues) - self.children[1].get(terminalValues)
        elif self.operation == '*':
            return self.children[0].get(terminalValues) * self.children[1].get(terminalValues)
        elif self.operation == '/':
            denom = self.children[1].get(terminalValues)
            if denom == 0:
                denom = 0.00001
            return self.children[0].get(terminalValues) / denom

        elif self.operation == 'step':
            if self.children[0].get(terminalValues) >= self.children[1].get(terminalValues):
                return 1
            else:
                return 0

        elif self.operation in GPNode.dataTerminals:
            return terminalValues[self.operation]

        elif self.operation == 'constant':
            return self.data

        elif self.operation == 'random':
            return random.expovariate(0.07)

    def getString(self):
        if self.operation in GPNode.nonTerminals:
            result = "(" + self.children[0].getString() + " " + self.operation + " " + self.children[1].getString() + ")"
        elif self.operation == 'constant':
            result = str(self.data)
        else:
            result = self.operation
        return result

    def getCode(self):
        # Returns executable python code for getting the selection chance from a population member p
        result = ''
        if self.operation in GPNode.nonTerminals:
            if len(self.children) == 2:
                if self.operation in ('+', '-', '*'):
                    result = '(' + self.children[0].getCode() + self.operation + self.children[1].getCode() + ')'
                elif self.operation == '/':
                    result = '(' + self.children[0].getCode() + self.operation + '(0.000001+' + self.children[1].getCode() + '))'
                elif self.operation == 'step':
                    result = '(int(' + self.children[0].getCode() + '>=' + self.children[1].getCode() + '))'

        elif self.operation in GPNode.dataTerminals:
            result = '(p.' + self.operation + ')'

        elif self.operation == 'constant':
            result = '(' + str(self.data) + ')'
        elif self.operation == 'random':
            result = '(random.expovariate(0.07))'
        return result

    def getAllNodes(self):
        nodes = []
        nodes.append(self)
        if self.children is not None:
            for c in self.children:
                nodes.extend(c.getAllNodes())
        return nodes

    def getAllNodesDepthLimited(self, depthLimit):
        # returns all nodes down to a certain depth limit
        nodes = []
        nodes.append(self)
        if self.children is not None and depthLimit > 0:
            for c in self.children:
                nodes.extend(c.getAllNodesDepthLimited(depthLimit - 1))
        return nodes

    def getDict(self):
        # return a dictionary containing the operation, data, and children of the node
        result = dict()
        result['data'] = self.data
        result['operation'] = self.operation
        if self.children is not None:
            result['children'] = []
            for c in self.children:
                result['children'].append(c.getDict())
        return result

    def buildFromDict(self, d):
        # builds a GPTree from a dictionary output by getDict
        self.data = d['data']
        self.operation = d['operation']
        if 'children' in d.keys():
            self.children = []
            for c in d['children']:
                newNode = GPNode()
                newNode.buildFromDict(c)
                newNode.parent = self
                self.children.append(newNode)
        else:
            self.children = None

class GPTree:
    # encapsulates a tree made of GPNodes that determine probability of selection, as well as other options relating
    # to parent selection
    def __init__(self):
        self.root = None
        self.fitness = None
        self.reusingParents = None
        self.final = False

    def roulette_selection(self, population, weights):
        # makes a random weighted selection from the population

        # raise an error if the lengths of the population and weights are different
        if len(population) != len(weights):
            raise IndexError

        # normalize the weights, if necessary
        normalized_weights = weights
        min_weight = min(weights)
        if min_weight < 0:
            for i in range(len(weights)):
                normalized_weights[i] -= min_weight

        # calculate the sum weight and select a number between 0 and the sum weight
        sum_weight = sum(normalized_weights)
        selection_number = random.uniform(0, sum_weight)

        # iterate through the items in the population until weights up to the selection number have passed, then return
        # the current item
        i = 0
        while selection_number > normalized_weights[i]:
            selection_number -= normalized_weights[i]
            i += 1

        return population[i]


    def recombine(self, parent2):
        # recombines two GPTrees and returns a new child

        # copy the first parent
        newChild = copy.deepcopy(self)
        newChild.fitness = None

        # select a point to insert a tree from the second parent
        insertionPoint = random.choice(newChild.getAllNodes())

        # copy a tree from the second parent
        replacementTree = copy.deepcopy(random.choice(parent2.getAllNodes()))

        # insert the tree
        newChild.replaceNode(insertionPoint, replacementTree)
        
        # recombine misc options
        newChild.reusingParents = random.choice([self.reusingParents, parent2.reusingParents])

        return newChild

    def mutate(self):
        # replaces a randomly selected subtree with a new random subtree, and flips the misc options

        # select a point to insert a new random tree
        insertionPoint = random.choice(self.getAllNodes())

        # randomly generate a new subtree
        newSubtree = GPNode()
        newSubtree.grow(3, None)

        # insert the new subtree
        self.replaceNode(insertionPoint, newSubtree)
        
        # 50/50 chance to flip misc option
        if random.random() < 0.5:
            self.reusingParents = not self.reusingParents
        
    def get(self, terminalValues):
        return self.root.get(terminalValues)
    
    def getAllNodes(self):
        result = self.root.getAllNodes()
        return result
    
    def getAllNodesDepthLimited(self, depthLimit):
        result = self.root.getAllNodesDepthLimited(depthLimit)
        return result
    
    def replaceNode(self, nodeToReplace, replacementNode):
        # replaces node in GPTree. Uses the replacementNode directly, not a copy of it
        if nodeToReplace is self.root:
            self.root = replacementNode
            self.root.parent = None
        else:
            parentOfReplacement = nodeToReplace.parent
            for i, c in enumerate(parentOfReplacement.children):
                if c is nodeToReplace:
                    parentOfReplacement.children[i] = replacementNode
                    break
            replacementNode.parent = parentOfReplacement

    def randomize(self, initialDepthLimit):
        if self.root is None:
            self.root = GPNode()
        self.root.grow(initialDepthLimit, None)
        
        self.reusingParents = bool(random.random() < 0.5)

    def verifyParents(self):
        for n in self.getAllNodes():
            if n is self.root:
                assert(n.parent is None)
            else:
                assert(n.parent is not None)
                assert(n in n.parent.children)

    def getString(self):
        return self.root.getString() + ' | reusing parents: {0}'.format(self.reusingParents)

    def getCode(self):
        return self.root.getCode()

    def getDict(self):
        return self.root.getDict()

    def buildFromDict(self, d):
        self.fitness = None
        self.root = GPNode()
        self.root.buildFromDict(d)

    def saveToDict(self, filename):
        with open(filename, 'wb') as pickleFile:
            pickle.dump(self.getDict(), pickleFile)

    def loadFromDict(self, filename):
        with open(filename, 'rb') as pickleFile:
            d = pickle.load(pickleFile)
            self.buildFromDict(d)

    def isClone(self, population):
        # returns true if this GPTree is a clone of any members of the given population
        # uses the getString() function of the GPTree, so there may be some false negatives, but no false positives
        for p in population:
            if self is not p and self.getString() == p.getString():
                return True
        return False

class Eppsea:
    def __init__(self, configPath=None):

        # set up the results directory
        presentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experimentName = "eppsea_" + str(presentTime)
        resultsDirectory = "./results/eppsea/" + experimentName
        os.makedirs(resultsDirectory)
        self.resultsDirectory = resultsDirectory

        # setup logging file
        self.logFile = open(self.resultsDirectory + '/log.txt', 'w')

        # try to read a config file from the config file path
        # if we do not have a config file, generate and use a default config
        if configPath is None:
            self.log('No config file path provided. Generating and using default config.', 'INFO')
            configPath = 'config/base_config/default.cfg'
            self.generateDefaultConfig(configPath)
        # if the provided file path does not exist, generate and use a default config
        elif not os.path.isfile(str(configPath)):
            self.log('No config file found at {0}. Generating and using default config.'.format(configPath), 'INFO')
            configPath = 'config/base_config/default.cfg'
            self.generateDefaultConfig(configPath)
        else:
            self.log('Using config file {0}'.format(configPath), 'INFO')

        # copy the used config file to the results path
        shutil.copyfile(configPath, resultsDirectory + '/config_used.cfg')

        # set up the configuration object
        config = configparser.ConfigParser()
        config.read(configPath)

        # get the parameters from the configuration file
        self.GPMu = config.getint('metaEA', 'metaEA mu')
        self.GPLambda = config.getint('metaEA', 'metaEA lambda')
        self.maxGPEvals = config.getint('metaEA', 'metaEA maximum fitness evaluations')
        self.initialGPDepthLimit = config.getint('metaEA', 'metaEA GP tree initialization depth limit')
        self.GPKTournamentK = config.getint('metaEA', 'metaEA k-tournament size')
        self.GPSurvivalSelection = config.get('metaEA', 'metaEA survival selection')

        self.terminateMaxEvals = config.getboolean('metaEA', 'terminate on maximum evals')
        self.terminateNoAvgFitnessChange = config.getboolean('metaEA', 'terminate on no improvement in average fitness')
        self.terminateNoBestFitnessChange = config.getboolean('metaEA', 'terminate on no improvement in best fitness')
        self.noChangeTerminationGenerations = config.get('metaEA', 'generations to termination for no improvement')

        self.restartNoAvgFitnessChange = config.getboolean('metaEA', 'restart on no improvement in average fitness')
        self.restartNoBestFitnessChange = config.getboolean('metaEA', 'restart on no improvement in best fitness')
        self.noChangeRestartGenerations = config.getint('metaEA', 'generations to restart for no improvement')

        self.GPmutationRate = config.getfloat('metaEA', 'metaEA mutation rate')
        self.forceMutationOfClones = config.getboolean('metaEA', 'force mutation of clones')

        self.pickleEveryPopulation = config.getboolean('experiment', 'pickle every population')
        self.pickleFinalPopulation = config.getboolean('experiment', 'pickle final population')

        # create a dictionary for the results
        self.results = dict()
        self.results['evalCounts'] = []
        self.results['averageFitness'] = []
        self.results['bestFitness'] = []

        # setup evolution data structures
        self.population = None
        self.new_population = None
        self.genNumber = None
        self.evolutionFinished = None
        self.gensSinceAvgFitnessImprovement = None
        self.gensSinceBestFitnessImprovement = None
        self.highestAverageFitness = None
        self.highestBestFitness = None
        self.GPEvals = None
        self.restarting = None
        self.finalBestMember = None

        self.startTime = None
        self.timeElapsed = None

        # seed RNG and record seed
        seed = config.get('experiment', 'seed')
        try:
            seed = int(seed)
        except ValueError:
            seed = int(time.time())
        random.seed(seed)
        self.log('Using random seed {0}'.format(seed), 'INFO')

    # record start time
    startTime = time.time()

    def log(self, message, messageType):
        # Builds a log message out of a timestamp, the passed message, and a message type, then prints the message and
        # writes it in the logFile

        # Get the timestamp
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Put together the full message
        fullMessage = '{0}| {1}: {2}'.format(timestamp, messageType, message)

        # Print and log the message
        print(fullMessage)
        self.logFile.write(fullMessage)
        self.logFile.write('\n')

    def startEvolution(self):
        self.log('Starting evolution', 'INFO')
        # record start time
        self.startTime = time.time()

        # initialize the population
        self.population = []
        for i in range(self.GPMu):
            newTree = GPTree()
            newTree.randomize(self.initialGPDepthLimit)
            self.population.append(newTree)

        # mark the entire population as new
        self.new_population = list(self.population)

        # check population uniqueness
        self.checkGPPopulationUniqueness(self.population, 0.75)

        # start evolution variables
        self.GPEvals = 0
        self.genNumber = 0
        self.evolutionFinished = False
        self.gensSinceAvgFitnessImprovement = 0
        self.gensSinceBestFitnessImprovement = 0
        self.highestAverageFitness = -math.inf
        self.highestBestFitness = -math.inf
        self.restarting = False

    def nextGeneration(self):
        # make sure all population members have been assigned fitness values
        if not all(p.fitness is not None for p in self.population):
            self.log('Attempting to advance to next generation before assigning fitness to all members', 'ERROR')
            return

        # log and increment generation
        self.log('Finished generation {0}'.format(self.genNumber), 'INFO')
        self.genNumber += 1

        # increment evaluation counter
        self.GPEvals += len(self.new_population)

        # Update results
        averageFitness = statistics.mean(p.fitness for p in self.population)
        bestFitness = max(p.fitness for p in self.population)

        self.results['evalCounts'].append(self.GPEvals)
        self.results['averageFitness'].append(averageFitness)
        self.results['bestFitness'].append(bestFitness)

        # pickle the population, if configured to
        if self.pickleEveryPopulation:
            pickleDirectory = self.resultsDirectory + '/pickledPopulations'
            pickleFilePath = pickleDirectory + '/gen{0}'.format(self.genNumber)
            os.makedirs(pickleDirectory, exist_ok=True)
            with open(pickleFilePath, 'wb') as pickleFile:
                pickle.dump(self.population, pickleFile)

        # check termination and restart conditions
        if averageFitness > self.highestAverageFitness:
            self.highestAverageFitness = averageFitness
            self.gensSinceAvgFitnessImprovement = 0
        else:
            self.gensSinceAvgFitnessImprovement += 1
            if self.terminateNoAvgFitnessChange and self.gensSinceAvgFitnessImprovement >= self.noChangeTerminationGenerations:
                self.log('Terminating evolution due to no improvement in average fitness', 'INFO')
                self.evolutionFinished = True
            elif self.restartNoAvgFitnessChange and self.gensSinceAvgFitnessImprovement >= self.noChangeRestartGenerations:
                self.log('Restarting evolution due to no improvement in average fitness', 'INFO')
                self.restarting = True
        if bestFitness > self.highestBestFitness:
            self.highestBestFitness = bestFitness
            self.gensSinceBestFitnessImprovement = 0
        else:
            self.gensSinceBestFitnessImprovement += 1
            if self.terminateNoBestFitnessChange and self.gensSinceBestFitnessImprovement >= self.noChangeTerminationGenerations:
                self.log('Terminating evolution due to no improvement in best fitness', 'INFO')
                self.evolutionFinished = True
            elif self.restartNoBestFitnessChange and self.gensSinceBestFitnessImprovement >= self.noChangeRestartGenerations:
                self.log('Restarting evolution due to no improvement in best fitness', 'INFO')
                self.restarting = True
        if self.terminateMaxEvals and self.GPEvals >= self.maxGPEvals:
            self.log('Terminating evolution due to max evaluations reached', 'INFO')
            self.evolutionFinished = True

        if not self.evolutionFinished:

            # if we are restarting, regenerate the population
            if self.restarting:
                self.population = []
                for i in range(self.GPMu):
                    newTree = GPTree()
                    newTree.randomize(self.initialGPDepthLimit)
                    self.population.append(newTree)
                self.new_population = list(self.population)

                self.gensSinceAvgFitnessImprovement = 0
                self.gensSinceBestFitnessImprovement = 0
                self.highestAverageFitness = -math.inf
                self.highestBestFitness = -math.inf
                self.restarting = False

            # otherwise, do survival selection and generate the next generation
            else:
                # survival selection
                if self.GPSurvivalSelection == 'random':
                    self.population = random.sample(self.population, self.GPMu)
                elif self.GPSurvivalSelection == 'truncation':
                    self.population.sort(key=lambda p: p.fitness, reverse=True)
                    self.population = self.population[:self.GPMu]

                # parent selection and new child generation
                self.new_population = []
                while len(self.new_population) < self.GPLambda:
                    # parent selection (k tournament)
                    parent1 = max(random.sample(self.population, self.GPKTournamentK), key=lambda p: p.fitness)
                    parent2 = max(random.sample(self.population, self.GPKTournamentK), key=lambda p: p.fitness)
                    # recombination/mutation
                    newChild = parent1.recombine(parent2)
                    if random.random() < self.GPmutationRate or (
                            self.forceMutationOfClones and newChild.isClone(self.population + self.new_population)):
                        newChild.mutate()
                    self.new_population.append(newChild)

                # extend population with new members
                self.population.extend(self.new_population)

        else:
            # pickle the final population, if configured to
            if self.pickleFinalPopulation:
                pickleDirectory = self.resultsDirectory + '/pickledPopulations'
                pickleFilePath = pickleDirectory + '/final'
                os.makedirs(pickleDirectory, exist_ok=True)
                with open(pickleFilePath, 'wb') as pickleFile:
                    pickle.dump(self.population, pickleFile)

            # write the results
            with open(self.resultsDirectory + '/results.csv', 'w') as resultFile:

                resultWriter = csv.writer(resultFile)

                resultWriter.writerow(['evals', 'average fitness', 'best fitness'])
                resultWriter.writerow(self.results['evalCounts'])
                resultWriter.writerow(self.results['averageFitness'])
                resultWriter.writerow(self.results['bestFitness'])

            # find the best population member, log its string, and expose it
            self.finalBestMember = max(self.population, key=lambda p: p.fitness)
            finalBestMemberString = self.finalBestMember.getString()
            self.log('String form of best Popi: {0}'.format(finalBestMemberString), 'INFO')

            # log time elapsed
            self.timeElapsed = time.time() - self.startTime
            self.log('Time elapsed: {0}'.format(self.timeElapsed), 'INFO')

        return

    def checkGPPopulationUniqueness(self, population, warningThreshold):
        populationStrings = list(p.getString() for p in population)
        uniqueStrings = set(populationStrings)
        uniqueness = len(uniqueStrings) / len(populationStrings)
        if uniqueness <= warningThreshold:
            self.log('GP population uniqueness is at {0}%. Consider increasing mutation rate.'.format(round(uniqueness*100)), 'WARNING')

    def generateDefaultConfig(self, filePath):
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
                '\n'
            ])
