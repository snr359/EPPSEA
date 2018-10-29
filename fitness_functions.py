import random
import itertools
import uuid
import configparser

try:
    import cocoex
except ImportError:
    print('BBOB COCO not found. COCO benchmarks will not be available')

class FitnessFunction:
    genome_types = {
        'nk_landscape': 'bool',
        'mk_landscape': 'bool',
        'coco': 'float'
    }

    def __init__(self):
        self.type = None
        self.display_name = None
        self.genome_type = None
        self.genome_length = None
        self.max_initial_range = None

        self.epistasis_k = None
        self.epistasis = None
        self.loci_values = None

        self.coco_function_index = None
        self.coco_function_id = None
        self.coco_function = None

        self.started = False

    def assign_id(self):
        # assigns a random id to self. Every unique Fitness Function should call this once
        self.id = '{0}_{1}_{2}'.format('FitnessFunction', str(id(self)), str(uuid.uuid4()))

    def generate(self, config):
        # performs all first-time generation for a fitness function. Should be called once per unique fitness function

        self.type = config.get('EA', 'fitness function')
        self.genome_type = self.genome_types.get(self.type, None)

        self.display_name = config.get('fitness function','display name')

        self.genome_length = config.getint('fitness function', 'genome length')
        self.max_initial_range = config.getfloat('fitness function', 'max initial range')
        self.trap_size = config.getint('fitness function', 'trap size')
        self.epistasis_k = config.getint('fitness function', 'epistasis k')
        self.epistasis_m = config.getint('fitness function', 'epistasis m')

        self.coco_function_index = config.getint('fitness function', 'coco function index')

        if self.type == 'nk_landscape':
            self.loci_values, self.epistasis = self.generate_nk_epistatis(self.genome_length, self.epistasis_k)
        elif self.type == 'mk_landscape':
            self.loci_values, self.epistasis = self.generate_mk_epistatis(self.genome_length, self.epistasis_m, self.epistasis_k)
        else:
            self.loci_values, self.epistasis = None, None

        self.assign_id()

    def start(self):
        # should be called once at the start of each search
        if self.type == 'coco':
            suite = cocoex.Suite('bbob', '', 'dimensions:{0}, function_indices:{1}'.format(self.genome_length, self.coco_function_index))
            self.coco_function = suite.get_problem(self.coco_function_id)
        self.started = True

    def finish(self):
        # should be called once at the end of each EA
        self.coco_function = None
        self.started = False

    def random_genome(self):
        # generates and returns a random genome

        # generate a new genome
        genome = []

        # generate values and append to the genome
        for _ in range(self.genome_length):
            if self.genome_type == 'bool':
                genome.append(random.choice((True, False)))
            elif self.genome_type == 'float':
                genome.append(random.uniform(-self.max_initial_range, self.max_initial_range))
            else:
                print('WARNING: genome type {0} not recognized for random genome generation'.format(self.genome_type))
                break

        # return the new genome
        return genome

    def fitness_target_hit(self):
        if self.type == 'coco':
            return self.coco_function.final_target_hit
        else:
            return False

    def nk_landscape(self, x):
        result = 0

        for i in range(self.genome_length):
            locus = [x[i]]
            locus.extend((x[j] for j in self.epistasis[i]))
            locus_fitness = self.loci_values[tuple(locus)]
            result += locus_fitness

        return result

    def mk_landscape(self, x):
        result = 0

        for e in self.epistasis:
            locus = []
            locus.extend(x[j] for j in e)
            locus_fitness = self.loci_values[tuple(locus)]
            result += locus_fitness

        return result

    def coco(self, x):
        # multiply by -1 since coco functions are minimization functions
        return -1 * self.coco_function(x)

    def generate_nk_epistatis(self, n, k):
        loci_values = dict()
        for locus in itertools.product([True, False], repeat=k+1):
            loci_values[locus] = random.randint(0,k)

        epistasis = dict()
        for i in range(n):
            epistasis[i] = sorted(random.sample(list(j for j in range(n) if j != i), k))

        return loci_values, epistasis

    def generate_mk_epistatis(self, n, m, k):
        loci_values = dict()
        for locus in itertools.product([True, False], repeat=k):
            loci_values[locus] = random.randint(0,k)

        epistasis = list()
        for _ in range(m):
            epistasis.append(list(random.sample(list(j for j in range(n)), k)))

        return loci_values, epistasis

    def evaluate(self, genome):
        if self.type == 'nk_landscape':
            fitness = self.nk_landscape(genome)
        elif self.type == 'mk_landscape':
            fitness = self.mk_landscape(genome)
        elif self.type == 'coco':
            fitness = self.coco(genome)

        else:
            raise Exception('EPPSEA BasicEA ERROR: fitness function name {0} not recognized'.format(self.type))

        return fitness

def generate_landscape_functions(config_path, n, append_instance_number):
    # generates and returns a list of n nk_landscape or mk_landscape functions
    # if append_instance_number, then the instance number will be apppended to the function's display name
    config = configparser.ConfigParser()
    config.read(config_path)

    fitness_functions = []

    for i in range(n):
        new_fitness_function = FitnessFunction()
        new_fitness_function.generate(config)

        if append_instance_number:
            new_fitness_function.display_name = '{0}_I{1}'.format(new_fitness_function.display_name, str(i))

        fitness_functions.append(new_fitness_function)

    return fitness_functions

def generate_coco_functions(config_path, append_instance_number):
    # given a path to a fitness function configuration file, generates all the coco fitness functions and returns them
    # if append_instance_number, then the instance number will be appended to the function's display name
    # generates the fitness functions to be used in the EAs

    config = configparser.ConfigParser()
    config.read(config_path)

    genome_length = config.getint('fitness function', 'genome length')
    if genome_length not in [2, 3, 5, 10, 20, 40]:
        print('WARNING: genome length {0} may not be supported by coco'.format(genome_length))
    coco_function_index = config.get('fitness function', 'coco function index')
    suite = cocoex.Suite('bbob', '','dimensions:{0}, function_indices:{1}'.format(genome_length, coco_function_index))
    coco_ids = list(suite.ids())

    fitness_functions = []
    for coco_id in coco_ids:
        new_fitness_function = FitnessFunction()
        new_fitness_function.generate(config)
        new_fitness_function.coco_function_id = coco_id

        if append_instance_number:
            new_fitness_function.display_name = '{0}_I{1}'.format(new_fitness_function.display_name, str(coco_id))

        fitness_functions.append(new_fitness_function)

    return fitness_functions

