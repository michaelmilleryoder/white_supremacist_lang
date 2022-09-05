""" Main entry point to run experiments:
        1. Load, process datasets, construct corpora
        2. Train and evaluate classifier for white supremacist language
"""

import yaml
import argparse
import pdb

import pandas as pd

from corpus import Corpus
#from bert_classifier import BertClassifier


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
    corpora = {key: Corpus(**opts).load() for key, opts in config['corpora'].items()}
    #corpora = {}
    #for role, info in config['corpora'].items():
    #    corpora[role] = Corpus(**info).load()

    # Train and evaluate classifier
    #clf = BertClassifier() # put in config options
    #clf.run()


if __name__ == '__main__':
    main()
