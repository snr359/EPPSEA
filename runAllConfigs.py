import sys
import subprocess
import os

# runs the eppsea_basicEA with all the configs in a given directory
config_directory = sys.argv[1]

pypyPath = '../pypy3-v5.10.1-linux64/bin/pypy3'

if os.path.exists('../pypy3-v5.10.1-linux64/bin/pypy3') and os.path.isfile('../pypy3-v5.10.1-linux64/bin/pypy3'):
    interpreter = '../pypy3-v5.10.1-linux64/bin/pypy3'
else:
    interpreter = 'python3'

for filename in os.listdir(config_directory):
    if filename.endswith('.cfg'):
        config_path = config_directory + '/' + filename
        subprocess.call([interpreter, 'eppsea_basicEA.py', config_path])