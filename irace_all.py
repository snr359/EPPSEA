# a quick script to run irace for all 24 coco functions

import configparser
import eppsea_basicEA
import sys
import subprocess

for i in range(1,25):
    eppsea_basicea_config_path = 'config/basicEA/config5_f{0}_d10.cfg'.format(i)
    eppsea_basicea_config = configparser.ConfigParser()
    eppsea_basicea_config.read(eppsea_basicea_config_path)

    # prepare fitness functions by initializing eppsea basicEA object
    eppsea_basicea = eppsea_basicEA.EppseaBasicEA(eppsea_basicea_config)

    # run irace
    irace_path = sys.argv[1]
    params = ['python3', 'init_irace.py', '-i', irace_path, '-c', eppsea_basicea_config_path, '-t', 'fitness_functions/coco_f{0}_d10/training'.format(i)]
    print(' '.join(params))
    subprocess.run(params)