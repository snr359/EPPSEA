# this script contains test functions for the objects in eppsea_base.py
# the EPPSEA object tested uses the config passed into sys.argv[1[

import eppsea_base
import math
import sys

class popi:
    # a dummy population member for testing population-based eppsea functions with
    def __init__(self, fitness=None):
        self.fitness = fitness


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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

    # TESTING "FITNESSRANK" TERMINAL -------------------------------------------------------
    print('Testing GPTree "fitnessRank" terminal')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
        'root': {
            'operation': 'fitnessRank',
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(sample_population))):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def test_gptree_other_terminals(num_trials=50000):
    pass

def test_gptree_special_cases(num_trials=50000):
    # TESTING "DIVIDE BY ZERO" SPECIAL_CASE -------------------------------------------------------
    print('Testing GPTree "divide" operator special case: division by 0')
    test_tree = eppsea_base.GPTree()
    test_tree.build_from_dict({
        'reusing_parents': True,
        'selection_type': 'proportional',
        'select_from_subset': False,
        'selection_subset_size': 2,
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(special_case_sample_population))):
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

    if any(abs(expected_proportions[i] - actual_proportions[i]) > 0.01 for i in range(len(special_case_sample_population))):
        print('TEST FAILED')
        print('Expected proportions: {0}'.format(expected_proportions))
        print('Actual proportions: {0}'.format(actual_proportions))
    else:
        print('TEST PASSED')

def main():
    test_gptree_fitness_terminals()
    test_gptree_operators()
    test_gptree_other_terminals()
    test_gptree_special_cases()


if __name__ == '__main__':
    main()