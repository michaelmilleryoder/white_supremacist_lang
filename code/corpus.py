""" Defines Corpus class, which loads, processes, and holds a number of datasets into a uniform format
    for training

    @author Michael Miller Yoder
    @year 2022
"""

import pdb

import pandas as pd

from data import Dataset


class Corpus:
    """ Holds a collection of datasets with similar properties (white supremacist language or "neutral" language, for example)
        Load, process these datasets into a uniform format for building classifiers
    """

    def __init__(self, name: str, create: bool, datasets: list = []):
        """ Args:
                name: name for the corpus
                create: whether to recreate the corpus by loading and processing each dataset
                        and saving to self.fpath.
                        If False, will attempt to load the corpus from self.fpath
                datasets: list of dictionaries with names and associated loading paths for datasets
        """
        self.name = name
        self.fpath = f'../data/{name}_corpus.json'
        self.create = create
        self.datasets = [Dataset(ds['name'], ds['source'], ds['domain'], ds['load_paths']) for ds in datasets]
        self.data = None

    def load(self):
        """ Load corpus by creating it (loading and processing datasets) or loading it from disk """
        dfs = []
        if self.create:
            for dataset in self.datasets:
                print(f"\tLoading and processing {dataset.name}...")
                dataset.load()
                dataset.process()
                dfs.append(dataset.data)
            self.data = pd.concat(dfs)
            self.save()
        else:
            # TODO: load processed corpora from disk
            self.data = None
        return self

    def save(self):
        """ Save out corpus data for easier loading """
        # Probs do a json thing (maybe json table)
        print(f"Saving data to {self.fpath}...")
        self.data.to_json(self.fpath, orient='table')
