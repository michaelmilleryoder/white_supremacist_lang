""" Defines Corpus class, which loads, processes, and holds a number of datasets into a uniform format
    for training

    @author Michael Miller Yoder
    @year 2022
"""

import os
import pdb

import pandas as pd
from tqdm import tqdm

from data import Dataset


class Corpus:
    """ Holds a collection of datasets with similar properties (white supremacist language or "neutral" language, for example)
        Load, process these datasets into a uniform format for building classifiers
    """

    def __init__(self, name: str, create: bool, datasets: list = [], ref_corpora: list[str] = None):
        """ Args:
                name: name for the corpus
                create: whether to recreate the corpus by loading and processing each dataset
                        and saving to self.fpath.
                        If False, will attempt to load the corpus from self.fpath
                datasets: list of dictionaries with names and associated loading paths for datasets
                ref_corpora: a list of the names of any reference corpora that are used to construct this corpus. 
                        Will be loaded from disk (must already be saved out) if create is True
        """
        self.name = name
        self.base_fpath = '../data/corpora/{}_corpus.json'
        self.base_tmp_fpath = '../tmp/{}_corpus.pkl'
        self.fpath = self.base_fpath.format(self.name)
        self.tmp_fpath = self.base_tmp_fpath.format(self.name) # pickling for faster loading
        self.create = create
        self.ref_corpora = ref_corpora
        ref_corpora = None
        if self.ref_corpora is not None and self.create:
            # Load reference corpora
            ref_corpora = {}
            for corpus_name in self.ref_corpora:
                ref_corpus_fpath = self.base_tmp_fpath.format(corpus_name)
                tqdm.write(f"\tLoading reference corpus {corpus_name}...")
                ref_corpora[corpus_name] = self.load_corpus(ref_corpus_fpath)
        self.datasets = [Dataset(
                ds['name'], ds['source'], ds['domain'], ds['load_paths'], ref_corpora=ref_corpora) for ds in datasets]
        self.data = None

    def load(self):
        """ Load corpus by creating it (loading and processing datasets) or loading it from disk """
        dfs = []
        if self.create:
            for dataset in self.datasets:
                print(f"\tLoading and processing {dataset.name} ({dataset.source})...")
                dataset.load()
                dataset.process()
                if self.ref_corpora is not None:
                    dataset.print_stats()
                dfs.append(dataset.data)
            self.data = pd.concat(dfs)
            # TODO: Save out stats on the dataset in a log or output dir (#posts, #words per domain type, overall)
            self.save()
        else:
            # Try loading from pickle since it's faster
            if os.path.exists(self.tmp_fpath):
                load_path = self.tmp_fpath
            else:
                load_path = self.fpath
            print(f"Loading corpus from {load_path}...")
            self.data = self.load_corpus(load_path)
        return self

    def save(self):
        """ Save out corpus data for easier loading """
        print(f"Saving corpus to {self.fpath}...")
        self.data.to_json(self.fpath, orient='table', indent=4)
        print(f"Saving corpus to {self.tmp_fpath}...") # for faster loading as an option
        self.data.to_pickle(self.tmp_fpath)

    @classmethod
    def load_corpus(cls, path):
        """ Load a corpus from disk and return it as a dataframe"""
        res = None
        if path.endswith('.json'):
            res = pd.read_json(path, orient='table')
        elif path.endswith('.pkl'):
            res = pd.read_pickle(path)
        return res
