import subprocess

for survival_selection in ['random', 'truncation']:
    for k_size in [4,6,8]:
        for mutation_rate in [0.1, 0.25, 0.5, 1.0]:
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
                            'metaEA GP tree initialization depth limit: 3\n',
                            'metaEA mutation rate: {0}\n'.format(mutation_rate),
                            'metaEA survival selection: truncation\n',
                            'force mutation of clones: True\n',
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
                    'runs: 20\n',
                    'fitness function: rosenbrock\n',
                    'base eppsea config path: newBaseConfig.cfg\n',

                    '[fitness function]\n',
                    'genome length: 10\n',
                    'a: 100\n',
                    'max initial range: 5\n',
                    'trap size: 0\n',
                    'epistasis k: 0\n',
                ])

            subprocess.call(['python', 'eppsea_basicEA.py', 'newBasicConfig.cfg'])