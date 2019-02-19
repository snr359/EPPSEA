# this script contains test functions for the objects in eppsea_base.py
# These tests are probably not comprehensive and could use some improvement

import eppsea_base
import math

class popi:
    # a dummy population member for testing population-based eppsea functions with
    def __init__(self, fitness=None):
        self.genome = [0,1,2,3]
        self.fitness = fitness
        self.birth_generation = 0

def within_tolerance(observed, expected):
    # returns true if the observed proportions are very close to the actual proportions
    if any(abs(o - e) > 0.01 for o,e in zip(observed, expected)):
        return False
    else:
        return True

def test_gptree_fitness_terminals(num_trials=50000):
    # first, build a sample population
    sample_population = [popi(5),
                         popi(0),
                         popi(15),
                         popi(10)]

    # TESTING "FITNESS" TERMINAL -------------------------------------------------------
    print('Testing GPTree "fitness" terminal')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'fitness',
            'data': None
        }
    })
    expected_proportions = [5 / 30, 0 / 30, 15 / 30, 10/ 30]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "RELATIVE_FITNESS" TERMINAL -------------------------------------------------------
    print('Testing GPTree "relative_fitness" terminal')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'relative_fitness',
            'data': None
        }
    })
    expected_proportions = [5 / 30, 0 / 30, 15 / 30, 10/ 30]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "FITNESSRANK" TERMINAL -------------------------------------------------------
    print('Testing GPTree "fitness_rank" terminal')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'fitness_rank',
            'data': None
        }
    })
    expected_proportions = [2 / 10, 1 / 10, 4 / 10, 3 / 10]

    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def test_gptree_operators(num_trials=50000):
    # first, build a sample population
    sample_population = [popi(5),
                         popi(20),
                         popi(10),
                         popi(15)]

    # TESTING "PLUS" OPERATOR -------------------------------------------------------
    print('Testing GPTree "plus" operator')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '+',
            'data': None,
            'children':[
                {
                    'operation': 'fitness',
                    'data': None
                },
                {
                    'operation': 'fitness',
                    'data': None
                }
            ]
        }
    })
    total = (5+5) + (20+20) + (10+10) + (15+15)
    expected_proportions = [(5+5) / total, (20+20) / total, (10+10) / total, (15+15) / total]

    selection_counts = [0] * len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "MULTIPLY" OPERATOR -------------------------------------------------------
    print('Testing GPTree "multiply" operator')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '*',
            'data': None,
            'children': [
                {
                    'operation': 'fitness',
                    'data': None
                },
                {
                    'operation': 'fitness',
                    'data': None
                }
            ]
        }
    })
    total = (5*5) + (20*20) + (10*10) + (15*15)
    expected_proportions = [(5*5) / total, (20*20) / total, (10*10) / total, (15*15) / total]

    selection_counts = [0] * len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "MINUS" OPERATOR -------------------------------------------------------
    print('Testing GPTree "minus" operator')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '-',
            'data': None,
            'children': [
                {
                    'operation': '*',
                    'data': None,
                    'children': [
                        {
                            'operation': 'fitness',
                            'data': None
                        },
                        {
                            'operation': 'fitness',
                            'data': None
                        }
                    ]
                },
                {
                    'operation': 'fitness',
                    'data': None
                }
            ]
        }
    })
    total = (5*5-5) + (20*20-20) + (10*10-10) + (15*15-15)
    expected_proportions = [(5*5-5) / total, (20*20-20) / total, (10*10-10) / total, (15*15-15) / total]

    selection_counts = [0] * len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "DIVIDE" OPERATOR -------------------------------------------------------
    print('Testing GPTree "divide" operator')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '/',
            'data': None,
            'children': [
                {
                    'operation': '+',
                    'data': None,
                    'children': [
                        {
                            'operation': 'fitness',
                            'data': None
                        },
                        {
                            'operation': 'fitness',
                            'data': None
                        }
                    ]
                },
                {
                    'operation': '*',
                    'data': None,
                    'children': [
                        {
                            'operation': 'fitness',
                            'data': None
                        },
                        {
                            'operation': 'fitness',
                            'data': None
                        }
                    ]
                }
            ]
        }
    })
    total = ((5+5) / (5*5)) + ((20+20) / (20*20)) + ((10+10) / (10*10)) + ((15+15) / (15*15))
    expected_proportions = [((5+5) / (5*5)) / total, ((20+20) / (20*20)) / total,  ((10+10) / (10*10)) / total, ((15+15) / (15*15)) / total]

    selection_counts = [0] * len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "STEP" OPERATOR -------------------------------------------------------
    print('Testing GPTree "step" operator')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'step',
            'data': None,
            'children':[
                {
                    'operation': 'fitness',
                    'data': None
                },
                {
                    'operation': 'constant',
                    'data': 12
                }
            ]
        }
    })
    expected_proportions = [0, 0.5, 0, 0.5]

    selection_counts = [0] * len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def test_gptree_other_terminals(num_trials=50000):
    # first, build a sample population
    sample_population = [popi(5),
                         popi(20),
                         popi(10),
                         popi(15)]

    # TESTING "MIN_FITNESS" TERMINAL -------------------------------------------------------
    print('Testing GPTree "min_fitness" terminal, test #1')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'min_fitness',
            'data': None
        }
    })
    expected_proportions = [1/4, 1/4, 1/4, 1/4]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    print('Testing GPTree "min_fitness" terminal, test #2')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '+',
            'data': None,
            'children':[
                {
                    'operation': 'fitness',
                    'data': 1
                },
                {
                    'operation': 'min_fitness',
                    'data': None
                }
            ]
        }
    })
    expected_proportions = [10 / 70, 25 / 70, 15 / 70, 20 / 70]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "MAX_FITNESS" TERMINAL -------------------------------------------------------
    print('Testing GPTree "max_fitness" terminal, test #1')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'max_fitness',
            'data': None
        }
    })
    expected_proportions = [1/4, 1/4, 1/4, 1/4]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    print('Testing GPTree "max_fitness" terminal, test #2')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '+',
            'data': None,
            'children':[
                {
                    'operation': 'fitness',
                    'data': 1
                },
                {
                    'operation': 'max_fitness',
                    'data': None
                }
            ]
        }
    })
    expected_proportions = [25 / 130, 40 / 130, 30 / 130, 35 / 130]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "BIRTH_GEN" TERMINAL -------------------------------------------------------
    print('Testing GPTree "birth_generation" terminal')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'birth_generation',
            'data': None,
            'children': []
        }
    })

    sample_population[0].birth_generation = 0
    sample_population[1].birth_generation = 1
    sample_population[2].birth_generation = 2
    sample_population[3].birth_generation = 3

    expected_proportions = [0 / 6, 1 / 6, 2 / 6, 3 / 6]
    selection_counts = [0]*len(sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(sample_population, 1)[0]
        selected_index = sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def test_gptree_special_cases(num_trials=50000):
    # TESTING "DIVIDE BY ZERO" SPECIAL_CASE -------------------------------------------------------
    print('Testing GPTree "divide" operator special case: division by 0')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': '/',
            'data': None,
            'children':[
                {
                    'operation': 'constant',
                    'data': 1
                },
                {
                    'operation': 'fitness',
                    'data': None
                }
            ]
        }
    })
    expected_proportions = [1, 0, 0, 0]
    special_case_sample_population = [popi(0),
                                      popi(1),
                                      popi(1),
                                      popi(1)]

    selection_counts = [0] * len(special_case_sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(special_case_sample_population, 1)[0]
        selected_index = special_case_sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING VERY SMALL FITNESSES SPECIAL_CASE -------------------------------------------------------
    print('Testing GPTree fitness operator special case: small fitnesses')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'fitness',
            'data': None
        }
    })
    expected_proportions = [0.25, 0.25, 0.25, 0.25]
    special_case_sample_population = [popi(float('1e-300')),
                         popi(float('1e-300')),
                         popi(float('1e-300')),
                         popi(float('1e-300'))]

    selection_counts = [0] * len(special_case_sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(special_case_sample_population, 1)[0]
        selected_index = special_case_sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING 0 FITNESSES SPECIAL_CASE -------------------------------------------------------
    print('Testing GPTree fitness operator special case: zero fitnesses')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'fitness',
            'data': None
        }
    })
    expected_proportions = [0.25, 0.25, 0.25, 0.25]
    special_case_sample_population = [popi(0),
                         popi(0),
                         popi(0),
                         popi(0)]

    selection_counts = [0] * len(special_case_sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(special_case_sample_population, 1)[0]
        selected_index = special_case_sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING INFINITY FITNESS SPECIAL_CASE -------------------------------------------------------
    print('Testing GPTree fitness operator special case: infinity fitness')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'constant_min': 0,
        'constant_max': 1,
        'random_min': 2,
        'random_max': 3,
        'id': 0,
        'root': {
            'operation': 'fitness',
            'data': None
        }
    })
    expected_proportions = [0.25, 0.25, 0.25, 0.25]
    special_case_sample_population = [popi(0),
                         popi(0),
                         popi(math.inf),
                         popi(0)]

    selection_counts = [0] * len(special_case_sample_population)
    for _ in range(num_trials):
        selected_individual = test_tree.select(special_case_sample_population, 1)[0]
        selected_index = special_case_sample_population.index(selected_individual)
        selection_counts[selected_index] += 1

    actual_proportions = list(s / num_trials for s in selection_counts)

    if not within_tolerance(actual_proportions, expected_proportions):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def main():
    #test_gptree_fitness_terminals()
    #test_gptree_operators()
    test_gptree_other_terminals()
    test_gptree_special_cases()


if __name__ == '__main__':
    main()