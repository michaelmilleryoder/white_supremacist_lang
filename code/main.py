""" Main entry point to run experiments:
        1. Load, process datasets, construct corpora
        2. Train and evaluate classifier for white supremacist language
"""

import yaml
import argparse
import pdb

import pandas as pd

from corpus import Corpus
from experiment import Experiment


def main():
    """ Run experiments """

    # Load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', nargs='?', type=str, help='file path to config file')
    args = parser.parse_args()
    with open(args.config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Consruct or load corpora
    print('Constructing or loading corpora...')
    corpora = {key: Corpus(key, **opts).load() for key, opts in config['corpora'].items()}
    #corpora = {key: Corpus(key, **opts).load() for key, opts in config['corpora'].items() if opts['create']} # for debugging
    #corpora = [Corpus(**opts).load() for opts in config['corpora']]
    #corpora = [Corpus(**opts).load() for opts in config['corpora'] if opts['create']] # for debugging
    #corpora = {}
    #for role, info in config['corpora'].items():
    #    corpora[role] = Corpus(**info).load()

    # Train and evaluate classifier
    if config['experiment']['train'] or config['experiment']['test']:
        exp = Experiment(
                        config['experiment']['name'],
                        config['experiment']['train'],
                        config['experiment']['test'],
                        corpora[config['experiment']['train_pos']], 
                        corpora[config['experiment']['train_neg']], 
                        [corpora[corpus_name] for corpus_name in config['experiment']['test_corpora']], 
                        config['experiment']['test_by_dataset'],
                        config['experiment']['classifier'])
        exp.run()


if __name__ == '__main__':
    main()
