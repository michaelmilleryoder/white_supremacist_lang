""" Defines Corpus class, which loads, processes, and holds a number of datasets into a uniform format
    for training

    @author Michael Miller Yoder
    @year 2022
"""

import pdb

import pandas as pd
from tqdm import tqdm

from data import Dataset


class Corpus:
    """ Holds a collection of datasets with similar properties (white supremacist language or "neutral" language, for example)
        Load, process these datasets into a uniform format for building classifiers
    """

    def __init__(self, name: str, create: bool, datasets: list = [], ref_corpus_name: str = None):
        """ Args:
                name: name for the corpus
                create: whether to recreate the corpus by loading and processing each dataset
                        and saving to self.fpath.
                        If False, will attempt to load the corpus from self.fpath
                datasets: list of dictionaries with names and associated loading paths for datasets
                ref_corpus_name: the name of the reference corpus that is used to construct this corpus. 
                        Will be loaded from disk (must already be saved out) if create is True
        """
        self.name = name
        self.base_fpath = '../data/corpora/{}_corpus.json'
        self.base_tmp_fpath = '../tmp/{}_corpus.pkl'
        self.fpath = self.base_fpath.format(self.name)
        self.tmp_fpath = self.base_tmp_fpath.format(self.name) # pickling for faster loading
        self.create = create
        self.ref_corpus_name = ref_corpus_name
        ref_corpus = None
        if self.ref_corpus_name is not None and self.create:
            # Load reference corpus
            ref_corpus_fpath = self.base_fpath.format(self.ref_corpus_name)
            tqdm.write("\tLoading reference corpus...")
            ref_corpus = self.load_corpus(ref_corpus_fpath)
        self.datasets = [Dataset(
                ds['name'], ds['source'], ds['domain'], ds['load_paths'], ref_corpus=ref_corpus) for ds in datasets]
        self.data = None

    def load(self):
        """ Load corpus by creating it (loading and processing datasets) or loading it from disk """
        dfs = []
        if self.create:
            for dataset in self.datasets:
                print(f"\tLoading and processing {dataset.name} ({dataset.source})...")
                dataset.load()
                dataset.process()
                if self.ref_corpus_name is not None:
                    dataset.print_stats()
                dfs.append(dataset.data)
            self.data = pd.concat(dfs)
            self.save()
        else:
            print(f"Loading corpus from {self.fpath}...")
            self.data = self.load_corpus(self.fpath)
        return self

    def save(self):
        """ Save out corpus data for easier loading """
        print(f"Saving data to {self.fpath}...")
        self.data.to_json(self.fpath, orient='table', indent=4)
        print(f"Saving data to {self.tmp_fpath}...") # for faster loading as an option
        self.data.to_pickle(self.tmp_fpath)

    @classmethod
    def load_corpus(cls, path):
        """ Load a corpus from disk and return it as a dataframe"""
        return pd.read_json(path, orient='table')
