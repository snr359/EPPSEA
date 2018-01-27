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

import numpy as np

import basicEA

class GPNode:
    numericTerminals = ['constant'] #TODO: include random later?
    dataTerminals = ['fitness', 'fitnessProportion', 'fitnessRank', 'populationSize']
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

        else:
            log('operation {0} not found in value getter'.format(str(self.operation)), 'ERROR', logFile)

    def getString(self):
        if self.operation in GPNode.nonTerminals:
            if len(self.children) == 2:
                result = "(" + self.children[0].getString() + " " + self.operation + " " + self.children[1].getString() + ")"
            else:
                log('Nonterminal GP node with {0} children'.format(len(self.children)), 'WARNING', logFile)
                result = ''
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

            else:
                log('Nonterminal GP node with {0} children'.format(len(self.children)), 'WARNING', logFile)

        elif self.operation in GPNode.dataTerminals:
            result = '(p.' + self.operation + ')'

        elif self.operation == 'constant':
            result = '(' + str(self.data) + ')'
        elif self.operation == 'random':
            result = '(random.expovariate(0.07))'
        else:
            log('Operation {0} not found in code string generation'.format(str(self.operation)), 'ERROR', logFile)
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
        if nodeToReplace not in self.getAllNodes():
            log('Attempting to replace node not in own tree', 'ERROR', logFile)
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

def log(message, messageType, logFile):
    # Builds a log message out of a timestamp, the passed message, and a message type, then prints the message and
    # writes it in the logFile

    # Get the timestamp
    timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Put together the full message
    fullMessage = '{0}| {1}: {2}'.format(timestamp, messageType, message)

    # Print and log the message
    print(fullMessage)
    logFile.write(fullMessage)
    logFile.write('\n')

def evaluateGPPopulation(population):
    usingMultiprocessing = getFromConfig('experiment', 'use multiprocessing', 'bool')
    leaveOneCore = getFromConfig('experiment', 'leave one core idle', 'bool')
    numRuns = 5

    for p in population:
        if usingMultiprocessing:
            numProcesses = getFromConfig('experiment', 'processes', 'int')
            if numProcesses <= 0:
                if leaveOneCore:
                    pool = multiprocessing.Pool(os.cpu_count() - 1)
                else:
                    pool = multiprocessing.Pool()
            else:
                pool = multiprocessing.Pool(processes=numProcesses)

            params = ['rosenbrock', 20, 5000, 100, 20, 0.1, p]
            results = pool.starmap(basicEA.one_run, [params]*numRuns)
        else:
            results = []
            for i in range(numRuns):
                results.append(basicEA.one_run('rosenbrock', 20, 5000, 100, 20, 0.1, p))

        average_best = statistics.mean(r['best_fitness'] for r in results)
        p.fitness = average_best

def eppsea():
    # runs the meta EA for one run

    # get the parameters from the configuration file
    GPMu = getFromConfig('metaEA', 'metaEA mu', 'int')
    GPLambda = getFromConfig('metaEA', 'metaEA lambda', 'int')
    maxGPEvals = getFromConfig('metaEA', 'metaEA maximum fitness evaluations', 'int')
    initialGPDepthLimit = getFromConfig('metaEA', 'metaEA GP tree initialization depth limit', 'int')
    GPKTournamentK = getFromConfig('metaEA', 'metaEA k-tournament size', 'int')

    terminateMaxEvals = getFromConfig('metaEA', 'terminate on maximum evals', 'bool')
    terminateNoAvgFitnessChange = getFromConfig('metaEA', 'terminate on no improvement in average fitness', 'bool')
    terminateNoBestFitnessChange = getFromConfig('metaEA', 'terminate on no improvement in best fitness', 'bool')
    noChangeTerminationGeneraitons = getFromConfig('metaEA', 'generations to termination for no improvement', 'int')

    restartNoAvgFitnessChange = getFromConfig('metaEA', 'restart on no improvement in average fitness', 'bool')
    restartNoBestFitnessChange = getFromConfig('metaEA', 'restart on no improvement in best fitness', 'bool')
    noChangeRestartGeneraitons = getFromConfig('metaEA', 'generations to restart for no improvement', 'int')

    GPmutationRate = getFromConfig('metaEA', 'metaEA mutation rate', 'float')

    pickleEveryPopulation = getFromConfig('experiment', 'pickle every population', 'bool')
    pickleFinalPopulation = getFromConfig('experiment', 'pickle final population', 'bool')

    # create a dictionary for the results
    results = dict()
    results['evalCounts'] = []
    results['averageFitness'] = []
    results['bestFitness'] = []

    # initialize the trees
    GPPopulation = []
    for i in range(GPMu):
        newTree = GPTree()
        newTree.randomize(initialGPDepthLimit)
        GPPopulation.append(newTree)

    # Evaluate the initial population
    log('Evaluating initial population', 'INFO', logFile)
    evaluateGPPopulation(GPPopulation)

    # initialize eval counter
    GPEvals = GPMu

    # Update results
    averageFitness = statistics.mean(p.fitness for p in GPPopulation)
    bestFitness = max(p.fitness for p in GPPopulation)

    results['evalCounts'].append(GPEvals)
    results['averageFitness'].append(averageFitness)
    results['bestFitness'].append(bestFitness)

    # pickle the initial population, if configured to
    if pickleEveryPopulation:
        pickleDirectory = resultsDirectory + '/pickledPopulations'
        pickleFilePath = pickleDirectory + '/initial'
        os.makedirs(pickleDirectory, exist_ok=True)
        with open(pickleFilePath, 'wb') as pickleFile:
            pickle.dump(GPPopulation, pickleFile)

    # GP EA loop
    genNumber = 1
    evolutionFinished = False
    gensSinceAvgFitnessImprovement = 0
    gensSinceBestFitnessImprovement = 0
    highestAverageFitness = -1 * float('inf')
    highestBestFitness = -1 * float('inf')
    restarting = False

    while not evolutionFinished:

        log('Starting generation {0}'.format(genNumber), 'INFO', logFile)

        children = []
        while len(children) < GPLambda:
            # parent selection (k tournament)
            parent1 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.fitness)
            parent2 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.fitness)
            # recombination/mutation
            newChild = parent1.recombine(parent2)
            if random.random() < GPmutationRate:
                newChild.mutate()
            # add to the population
            children.append(newChild)

        # Evaluate children
        evaluateGPPopulation(children)
        GPEvals += GPLambda

        # population merging
        GPPopulation.extend(children)

        # survival selection (truncation)
        GPPopulation.sort(key=lambda p: p.fitness, reverse=True)
        GPPopulation = GPPopulation[:GPMu]

        # pickle the population, if configured to
        if pickleEveryPopulation:
            pickleDirectory = resultsDirectory + '/pickledPopulations'
            pickleFilePath = pickleDirectory + '/gen{0}'.format(genNumber)
            os.makedirs(pickleDirectory, exist_ok=True)
            with open(pickleFilePath, 'wb') as pickleFile:
                pickle.dump(GPPopulation, pickleFile)

        # Update results
        averageFitness = statistics.mean(p.fitness for p in GPPopulation)
        bestFitness = max(p.fitness for p in GPPopulation)

        results['evalCounts'].append(GPEvals)
        results['averageFitness'].append(averageFitness)
        results['bestFitness'].append(bestFitness)

        # check termination conditions
        if averageFitness > highestAverageFitness:
            highestAverageFitness = averageFitness
            gensSinceAvgFitnessImprovement = 0
        else:
            gensSinceAvgFitnessImprovement += 1
            if terminateNoAvgFitnessChange and gensSinceAvgFitnessImprovement >= noChangeTerminationGeneraitons:
                log('Terminating evolution due to no improvement in average fitness', 'INFO', logFile)
                evolutionFinished = True
            elif restartNoAvgFitnessChange and gensSinceAvgFitnessImprovement >= noChangeRestartGeneraitons:
                log('Restarting evolution due to no improvement in average fitness', 'INFO', logFile)
                restarting = True

        if bestFitness > highestBestFitness:
            highestBestFitness = bestFitness
            gensSinceBestFitnessImprovement = 0
        else:
            gensSinceBestFitnessImprovement += 1
            if terminateNoBestFitnessChange and gensSinceBestFitnessImprovement >= noChangeTerminationGeneraitons:
                log('Terminating evolution due to no improvement in best fitness', 'INFO', logFile)
                evolutionFinished = True
            elif restartNoBestFitnessChange and gensSinceBestFitnessImprovement >= noChangeRestartGeneraitons:
                log('Restarting evolution due to no improvement in best fitness', 'INFO', logFile)
                restarting = True

        if terminateMaxEvals and GPEvals >= maxGPEvals:
            log('Terminating evolution due to max evaluations reached', 'INFO', logFile)
            evolutionFinished = True

        # if we are restarting, regenerate and reevaluate the population
        if restarting:
            GPPopulation = []
            for i in range(GPMu):
                newTree = GPTree()
                newTree.randomize(initialGPDepthLimit)
                GPPopulation.append(newTree)
            log('Evaluating restarted population', 'INFO', logFile)
            evaluateGPPopulation(GPPopulation)
            GPEvals += GPMu

            gensSinceAvgFitnessImprovement = 0
            gensSinceBestFitnessImprovement = 0
            highestAverageFitness = -1 * float('inf')
            highestBestFitness = -1 * float('inf')
            restarting = False

        genNumber += 1

    # run the best population member one more time and get the full results
    log('Running best population member', 'INFO', logFile)
    bestGPPopi = max(GPPopulation, key=lambda p: p.fitness)
    finalResults = str(basicEA.one_run('rosenbrock', 20, 5000, 100, 20, 0.1, bestGPPopi))
    log('Result of best population member run: {0}'.format(finalResults), 'INFO', logFile)

    # write the results
    with open(resultsDirectory + '/results.csv', 'w') as resultFile:

        resultWriter = csv.writer(resultFile)

        resultWriter.writerow(['evals', 'average fitness', 'best fitness'])
        resultWriter.writerow(results['evalCounts'])
        resultWriter.writerow(results['averageFitness'])
        resultWriter.writerow(results['bestFitness'])

    # log final fitness stats
    finalAverageFitness = averageFitness

    finalBestFitness = bestFitness

    log('Final Average Fitness: {0}'.format(finalAverageFitness), 'INFO', logFile)
    log('Final Best Fitness: {0}'.format(finalBestFitness), 'INFO', logFile)

    # calculate and log string for final best GP popi
    bestGPPopiString = bestGPPopi.getString()
    log('String form of best Popi: {0}'.format(bestGPPopiString), 'INFO', logFile)
    
    # pickle the final population, if configured to
    if pickleFinalPopulation:
        pickleDirectory = resultsDirectory + '/pickledPopulations'
        pickleFilePath = pickleDirectory + '/final'
        os.makedirs(pickleDirectory, exist_ok=True)
        with open(pickleFilePath, 'wb') as pickleFile:
            pickle.dump(GPPopulation, pickleFile)

    return

def generateDefaultConfig(filePath):
    # generates a default configuration file and writes it to filePath
    with open(filePath, 'w') as file:
        file.writelines([
            '[experiment]\n', 
            'GP initialization depth limit: 3\n',
            'seed: time\n',
            'use multiprocessing: True\n',
            'processes: -1\n',
            'leave one core idle: True\n',
            'pickle every population: True\n',
            'pickle final population: True\n',
            '\n',
            '[metaEA]\n',
            'metaEA mu: 20\n',
            'metaEA lambda: 10\n',
            'metaEA maximum fitness evaluations: 200\n',
            'metaEA k-tournament size: 8\n',
            'metaEA GP tree initialization depth limit: 3\n',
            'metaEA mutation rate: 0.01\n',
            'terminate on maximum evals: True\n',
            'terminate on no improvement in average fitness: True\n',
            'terminate on no improvement in best fitness: True\n',
            'generations to termination for no improvement: 5\n',
            'restart on no improvement in average fitness: False\n',
            'restart on no improvement in best fitness: False\n',
            'generations to restart for no improvement: 5\n',
            '\n'
        ])

def getFromConfig(section, value, dataType=None):
    if dataType == 'bool':
        return config.getboolean(section, value)
    elif dataType == 'int':
        return config.getint(section, value)
    elif dataType == 'float':
        return config.getfloat(section, value)
    else:
        return config.get(section, value)

if __name__ == "__main__":
    # set up the results directory
    presentTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experimentName = "eppsea_" + str(presentTime)
    resultsDirectory = "./results/" + experimentName
    os.makedirs(resultsDirectory)

    # Set up the logging file
    logFile = open(resultsDirectory + '/log.txt', 'w')

    # read sys.argv[1] as the config file path
    # if we do not have a config file, generate and use a default config
    if len(sys.argv) < 2:
        log('No config file path provided. Generating and using default config.', 'INFO', logFile)
        configPath = 'default.cfg'
        generateDefaultConfig(configPath)
    # if the provided file path does not exist, generate and use a default config
    elif not os.path.isfile(sys.argv[1]):
        log('No config file found at {0}. Generating and using default config.'.format(sys.argv[1]), 'INFO', logFile)
        configPath = 'default.cfg'
        generateDefaultConfig(configPath)
    else:
        configPath = sys.argv[1]
        log('Using config file {0}'.format(configPath), 'INFO', logFile)

    # copy the used config file to the results path
    shutil.copyfile(configPath, resultsDirectory + '/config.cfg')

    # set up the configuration object. This will be referenced by multiple functions and classes within this module
    config = configparser.ConfigParser()
    config.read(configPath)

    # seed RNGs and record seed
    seed = getFromConfig('experiment', 'seed')
    try:
        seed = int(seed)
    except ValueError:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    log('Using random seed {0}'.format(seed), 'INFO', logFile)

    # record start time
    startTime = time.time()

    # run experiment
    eppsea()

    # log time elapsed
    timeElapsed = time.time() - startTime
    log("Time elapsed: {0}".format(timeElapsed), 'INFO', logFile)