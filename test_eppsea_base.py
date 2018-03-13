# this script contains test functions for the objects in eppsea_base.py
# the EPPSEA object tested uses the config passed into sys.argv[1[

import eppsea_base
import math
import sys

class popi:
    # a dummy population member for testing population-based eppsea functions with
    def __init__(self):
        self.fitness = None


def test_gptree_terminals():
    num_trials = 10000

    # first, build a sample population
    sample_population = []
    for i in range(4):
        sample_population.append(popi())
        sample_population[i].fitness = i*5

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
    expected_proportions = []
    total = sum(p.fitness for p in sample_population)
    for p in sample_population:
        expected_proportions.append(p.fitness / total)

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
    expected_proportions = []
    total = sum((i+1) for i in range(len(sample_population)))
    for i in range(len(sample_population)):
        expected_proportions.append((i+1) / total)

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

def main():
    test_gptree_terminals()

if __name__ == '__main__':
    main()