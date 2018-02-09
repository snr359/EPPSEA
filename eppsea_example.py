import random

import eppsea_base

if __name__ == '__main__':
    eppsea = eppsea_base.Eppsea('config/base_config/test.cfg')

    eppsea.startEvolution()

    while not eppsea.evolutionFinished:
        for p in eppsea.new_population:
            p.fitness = random.random()
        eppsea.nextGeneration()