import random

import eppsea_base

if __name__ == '__main__':
    eppsea = eppsea_base.Eppsea('config/base_config/test.cfg')

    eppsea.start_evolution()

    while not eppsea.evolution_finished:
        for p in eppsea.new_population:
            p.fitness = random.random()
        eppsea.next_generation()