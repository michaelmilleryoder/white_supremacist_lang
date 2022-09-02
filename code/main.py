""" Main entry point to run experiments:
        1. Load, process datasets, construct corpora
        2. Train and evaluate classifier for white supremacist language
"""

import yaml
import argparse
import pdb

import pandas as pd

from corpus import Corpus


def main():
    """ Run experiments """

    # Load settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', nargs='?', type=str, help='file path to config file')
    args = parser.parse_args()
    with open(args.config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Consruct or load corpora
    corpora = {key: Corpus(**opts).load() for key, opts in config['corpora']}






    datasets = [Dataset(name, load_paths=opts) for name, opts in config['datasets'].items()]
    if config['load_datasets']:
        print("Loading datasets...")
        loader = DatasetsLoader(datasets)
        loader.load_datasets(reprocess=config['reprocess_datasets'])

    # Run with-heg/no-heg comparison
    if config['heg_comparison']['run']:
        heg_comparison = HegComparison(datasets, 
            create_splits=config['heg_comparison']['create_splits'], 
            hate_ratio=config['hate_ratio'],
            cv_runs=config['heg_comparison']['cv_runs'],
        )
        heg_comparison.run(config['classifier']['name'], config['classifier']['settings'])

    # Run identity split PCA
    if config['pca']['run']:
        identity_pca = IdentityPCA(datasets, config['classifier']['name'], config['classifier']['settings'],
            create_datasets=config['pca']['create_identity_datasets'], 
            hate_ratio=config['hate_ratio'], 
            combine=config['pca']['combine_datasets'],
            incremental=config['pca']['incremental'])
        identity_pca.run()


if __name__ == '__main__':
    main()
