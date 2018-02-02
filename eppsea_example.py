import random

import eppsea_base

class example_evaluator:
    def evaluate(self, selection_function):
        return random.random()

if __name__ == '__main__':
    new_evaluator = example_evaluator()
    eppsea_base.eppsea(new_evaluator, 'config/base_config/test.cfg')