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

    # Train and evaluate classifier
    if config['experiment']['train'] or config['experiment']['test']:
        exp = Experiment(
                        config['experiment']['name'],
                        config['experiment']['train'],
                        config['experiment']['test'],
                        corpora,
                        config['experiment']['train_corpora'],
                        config['experiment']['test_corpora'],
                        #[corpora[corpus_info['name'] for corpus_info in config['experiment']['train_corpora'].items()], 
                        #[corpora[corpus_name] for corpus_d in config['experiment']['test_corpora']], 
                        config['experiment']['classifier'],
                        test_label_combine=config['experiment'].get('test_label_combine', None)
                        )
        exp.run()


if __name__ == '__main__':
    main()
