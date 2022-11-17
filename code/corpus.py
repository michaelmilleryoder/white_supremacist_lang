""" Defines Corpus class, which loads, processes, and holds a number of datasets into a uniform format
    for training

    @author Michael Miller Yoder
    @year 2022
"""

import os
import pickle
import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from data import Dataset


class Corpus:
    """ Holds a collection of datasets with similar properties (white supremacist language or "neutral" language, for example)
        Load, process these datasets into a uniform format for building classifiers
    """

    def __init__(self, name: str, create: bool, datasets: list = [], ref_corpora: list[str] = None, 
                    min_word_limit: int = 1, label = None, lda_filter: dict = None, sample: tuple[str, int] = ('', -1)):
        """ Args:
                name: name for the corpus
                create: whether to recreate the corpus by loading and processing each dataset
                        and saving to self.fpath.
                        If False, will attempt to load the corpus from self.fpath
                datasets: list of dictionaries with names and associated loading paths for datasets
                ref_corpora: a list of the names of any reference corpora that are used to construct this corpus. 
                        Will be loaded from disk (must already be saved out) if create is True
                min_word_limit: Minimum word limit of posts. Put 1 to take all posts
                label: classification label to apply to this corpus (string), 
                        or mapping from dataset labels to corpus labels (dict)
                lda_filter: dict with information to remove posts that match certain topics in a trained LDA model.
                        Includes 'model' with path to the trained model (sklearn LatentDirichletAllocation object), 
                        'query' of posts to consider removing, 'exclude_topics' with a list of topics to exclude
                sample: whether to sample a portion of the full corpus. 
                        Tuple of (query to select data, n to sample from that queried data)
                        ('', -1) to take the full corpus
        """
        self.name = name
        self.base_dirpath = '../data/corpora'
        self.fpath = os.path.join(self.base_dirpath, f'{self.name}_corpus.json')
        self.stats_fpath = os.path.join(self.base_dirpath, f'{self.name}_stats.jsonl')
        self.base_tmp_fpath = '../tmp/{}_corpus.pkl' # pickling for faster loading
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
        self.label = label
        self.datasets = [Dataset(
                ds['name'], ds['source'], ds['domain'], ds['load_paths'], 
                ref_corpora=ref_corpora, min_word_limit=min_word_limit) for ds in datasets]
        self.lda_filter = lda_filter
        self.sample_query, self.sample_n = sample
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
            self.data = pd.concat(dfs).drop_duplicates(subset='text')
            if self.lda_filter is not None:
                self.filter_lda()
            if self.sample_n > 0:
                self.sample()
            self.print_save_stats()
            self.save()
        else:
            # Try loading from pickle since it's faster
            if os.path.exists(self.tmp_fpath):
                load_path = self.tmp_fpath
            else:
                load_path = self.fpath
            print(f"Loading corpus from {load_path}...")
            self.data = self.load_corpus(load_path)
        # Assign labels
        if isinstance(self.label, str):
            self.data['label_str'] = self.label
        else: # is a dict mapping dataset labels to other labels
            self.data['label_str'] = self.data.label.map(self.label)
        return self

    def filter_lda(self):
        """ Remove posts that match certain topics in a trained LDA model """
        # Load trained model, vectorizer
        print("Filtering by LDA topics...")
        with open(self.lda_filter['model'], 'rb') as f:
            lda = pickle.load(f)
        with open(self.lda_filter['vectorizer'], 'rb') as f:
            vectorizer = pickle.load(f)
        selected = self.data.query(self.lda_filter['query']).copy()
        rest = self.data[~self.data.index.isin(selected.index)]

        # Infer topics on documents
        bow = vectorizer.transform(selected.text)
        doc_topics = lda.transform(bow)
        assigned_topics = np.argmax(doc_topics, axis=1)
        
        # Filter out documents with certain topics
        selected['topic'] = assigned_topics
        filtered = selected[~selected.topic.isin(self.lda_filter['exclude_topics'])]
        self.data = pd.concat([filtered.drop(columns='topic'), rest]).sort_index()

    def sample(self):
        """ Sample particular portions of the corpus """
        selected = self.data.query(self.sample_query)
        rest = self.data[~self.data.index.isin(selected.index)]
        sampled = selected.sample(self.sample_n)
        self.data = pd.concat([sampled, rest]).sort_index()

    def print_save_stats(self):
        """ Print, save out stats on the dataset in a log or output dir 
            (#posts, #words per domain type, overall)
        """
        self.data['num_words'] = self.data.text.str.split().str.len()
        stats = self.data.groupby('domain').agg({'num_words': ['count', 'sum', 'mean']})
        stats.columns = ['post_count', 'word_count', 'avg_post_words']
        total = pd.DataFrame({'post_count': len(self.data), 'word_count': self.data.num_words.sum(),
            'avg_post_words': self.data.num_words.mean()}, index=['total'])
        stats = pd.concat([stats,total])
        stats['post%'] = stats['post_count']/stats.loc['total', 'post_count']
        stats['word%'] = stats['word_count']/stats.loc['total', 'word_count']
        stats = stats[['post_count', 'post%', 'word_count', 'word%', 'avg_post_words']]
        stats.index.name = 'domain'
        print(f"Corpus {self.name} stats:")
        print(stats)
        stats.to_json(self.stats_fpath, orient='records', lines=True)

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
