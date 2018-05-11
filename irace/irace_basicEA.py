# This script runs irace (http://iridia.ulb.ac.be/irace/) to find the optimal parameters for the EA used in
# the eppsea_basicEA script. It takes as a paramater an eppsea_basicEA configuration, from which it determines
# some EA settings, and uses irace to find values for other settings.

import argparse

def get_args():
    # parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--population_size', type=int)
    parser.add_argument('--offspring_size', type=int)
    parser.add_argument('--mutation_rate', type=float)
    parser.add_argument('--survival_selection')
    parser.add_argument('--parent_selection')
    parser.add_argument('--parent_selection_tournament_k')



def main():
    # get the command-line arguments
    args = get_args()

if __name__ == '__main__':
    main()