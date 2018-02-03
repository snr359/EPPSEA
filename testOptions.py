import subprocess

for k_size in [4,6,8]:
    for init_depth in [3, 5]:
        for force_mutate in [True, False]:
            for mutation_rate in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
                with open('newBaseConfig.cfg', 'w') as cfg:
                    cfg.writelines(['[experiment]\n',
                                'GP initialization depth limit: 3\n',
                                'seed: time\n',
                                'use multiprocessing: True\n',
                                'processes: -1\n',
                                'leave one core idle: False\n',
                                'pickle every population: True\n',
                                'pickle final population: True\n',

                                '[metaEA]\n',
                                'metaEA mu: 20\n',
                                'metaEA lambda: 20\n',
                                'metaEA maximum fitness evaluations: 500\n',
                                'metaEA k-tournament size: {0}\n'.format(k_size),
                                'metaEA GP tree initialization depth limit: {0}\n'.format(init_depth),
                                'metaEA mutation rate: {0}\n'.format(mutation_rate),
                                'force mutation of clones: {0}\n'.format(force_mutate),
                                'terminate on maximum evals: True\n',
                                'terminate on no improvement in average fitness: False\n',
                                'terminate on no improvement in best fitness: False\n',
                                'generations to termination for no improvement: 5\n',
                                'restart on no improvement in average fitness: False\n',
                                'restart on no improvement in best fitness: False\n',
                                'generations to restart for no improvement: 5\n'])

                with open('newBasicConfig.cfg', 'w') as cfg2:
                    cfg2.writelines([
                        '[EA]\n',
                        'population size: 100\n',
                        'offspring size: 20\n',
                        'maximum evaluations: 5000\n',
                        'mutation rate: 0.05\n',
                        'runs: 15\n',
                        'fitness function: rosenbrock\n',
                        'base eppsea config path: newBaseConfig.cfg\n',

                        '[fitness function]\n',
                        'genome length: 20\n',
                        'a: 100\n',
                        'max initial range: 5\n',
                        'trap size: 0\n',
                        'epistasis k: 0\n',
                    ])

                subprocess.call(['python', 'eppsea_basicEA.py', 'newBasicConfig.cfg'])