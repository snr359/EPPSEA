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
    sample_population = [popi(0),
                         popi(5),
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
    expected_proportions = [0 / 30, 5 / 30, 15 / 30, 10/ 30]
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
    expected_proportions = [1 / 10, 2 / 10, 4 / 10, 3 / 10]

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
                         popi(10),
                         popi(15),
                         popi(20)]

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
    expected_proportions = []
    total = sum((p.fitness + p.fitness) for p in sample_population)
    for p in sample_population:
        expected_proportions.append((p.fitness + p.fitness) / total)

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
    expected_proportions = []
    total = sum((p.fitness * p.fitness) for p in sample_population)
    for p in sample_population:
        expected_proportions.append((p.fitness * p.fitness) / total)

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
    expected_proportions = []
    total = sum((p.fitness * p.fitness - p.fitness) for p in sample_population)
    for p in sample_population:
        expected_proportions.append(((p.fitness * p.fitness) - p.fitness) / total)

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
    expected_proportions = []
    total = sum(((p.fitness + p.fitness) / (p.fitness * p.fitness)) for p in sample_population)
    for p in sample_population:
        expected_proportions.append(((p.fitness + p.fitness) / (p.fitness * p.fitness)) / total)

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
    expected_proportions = [0, 0, 0.5, 0.5]

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

def test_gptree_other_terminals():
    pass

def main():
    test_gptree_fitness_terminals()
    test_gptree_operators()


if __name__ == '__main__':
    main()