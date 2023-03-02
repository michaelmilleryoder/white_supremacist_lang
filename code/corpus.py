""" Defines Corpus class, which loads, processes, and holds a number of datasets into a uniform format
    for training

    @author Michael Miller Yoder
    @year 2022
"""

import os
import pickle
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from data import Dataset


class Corpus:
    """ Holds a collection of datasets with similar properties (white supremacist language or "neutral" language, for example)
        Load, process these datasets into a uniform format for building classifiers
    """

    def __init__(self, name: str, create: bool, datasets: list = [], ref_corpora: list[str] = None, 
                    match_factor: int = 1, min_word_limit: int = 1, label = None, lda_filter: dict = None, 
                    sample: dict = None, remove_duplicates: bool = True, split: dict = None):
        """ Args:
                name: name for the corpus
                create: whether to recreate the corpus by loading and processing each dataset
                        and saving to self.fpath.
                        If False, will attempt to load the corpus from self.fpath
                datasets: list of dictionaries with names and associated info (loading paths, etc) for datasets
                ref_corpora: a list of the names of any reference corpora that are used to construct this corpus. 
                        Will be loaded from disk (must already be saved out) if create is True
                match_factor: factor to multiply even sample with reference corpus by
                min_word_limit: Minimum word limit of posts. Put 1 to take all posts
                label: classification label to apply to this corpus (string), 
                        or mapping from dataset labels to corpus labels (dict)
                lda_filter: dict with information to remove posts that match certain topics in a trained LDA model.
                        Includes 'model' with path to the trained model (sklearn LatentDirichletAllocation object), 
                        'query' of posts to consider removing, 'exclude_topics' with a list of topics to exclude
                sample: dict of info on sample a portion of the full corpus, or None if not sampling
                        query: to select data, 
                        sample_n: n to sample from that queried data
                        sample_factor: factor to multiply/sample the data by
                remove_duplicates: whether to remove duplicate texts in the corpus
                split: dict of info on train/test split. None if not splitting. 
                        Training and test splits (as well as the original) will be saved out.
                    Dict contains:
                        test_size: size (fraction) of the corpus to randomly sample as a test split
                        ref_dataset: split based on a prior dataset's train/test split
        """
        self.name = name
        self.base_dirpath = '../data/corpora'
        self.fpath = os.path.join(self.base_dirpath, f'{self.name}_corpus.json')
        self.stats_fpath = os.path.join(self.base_dirpath, f'{self.name}_stats.jsonl')
        self.base_tmp_fpath = '../tmp/{}_corpus.pkl' # pickling for faster loading
        self.tmp_fpath = self.base_tmp_fpath.format(self.name) # pickling for faster loading
        self.create = create
        self.ref_corpora = ref_corpora
        self.match_factor = match_factor
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
                ref_corpora=ref_corpora, match_factor=self.match_factor, min_word_limit=min_word_limit,
                include_users=ds.get('include_users', False), 
                remove_duplicates=ds.get('remove_duplicates', True)) for ds in datasets]
        self.lda_filter = lda_filter
        self.sample_info = sample
        self.remove_duplicates = remove_duplicates
        self.split = split
        self.split_ref = None
        if self.split is not None:
            self.train_suffix = f'_train{int((1-self.split["test_size"])*100)}'
            self.test_suffix = f'_test{int(self.split["test_size"]*100)}'
            if 'split_ref' in self.split:
                ref_corpus_fpath = self.base_tmp_fpath.format(self.split['split_ref'])
                self.split_ref = self.load_corpus(ref_corpus_fpath, split=True, 
                    train_suffix=self.train_suffix, test_suffix=self.test_suffix)
        self.folds = {}
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
            self.folds['all'] = pd.concat(dfs)
            if self.remove_duplicates:
                self.folds['all'] = self.folds['all'].drop_duplicates(subset='text')
            if self.lda_filter is not None:
                self.filter_lda()
            if self.split is not None:
                if 'split_ref' in self.split and self.split['split_ref'] is not None:
                    # Assign train/test labels based on the reference dataset
                    train = self.folds['all'][self.folds['all'].index.isin(self.split_ref['train'].index)]
                    test = self.folds['all'][self.folds['all'].index.isin(self.split_ref['test'].index)]
                    test_size = len(test)/(len(test) + len(train))
                    if test_size < self.split['test_size'] - 0.1 or test_size > self.split['test_size'] + 0.1:
                        pdb.set_trace() # reference train/test ratio is off from desired ratio
                    else:
                        self.folds['train'] = train
                        self.folds['test'] = test
                else:
                    self.folds['train'], self.folds['test'] = train_test_split(
                        self.folds['all'], test_size=self.split['test_size'], random_state=9)
            if self.sample_info is not None:
                self.sample()
            self.data = self.folds['all']
            self.print_save_stats()
            self.save()

        else:
            # Try loading from pickle since it's faster
            if os.path.exists(self.tmp_fpath):
                load_path = self.tmp_fpath
            else:
                load_path = self.fpath
            print(f"Loading corpus from {load_path}...")
            self.folds['all'] = self.load_corpus(load_path)
            if self.split is not None:
                path = Path(load_path)
                train_path = str(path.with_stem(str(path.stem) + self.train_suffix))
                test_path = str(path.with_stem(str(path.stem) + self.test_suffix))
                self.folds['train'], self.folds['test'] = self.load_corpus(train_path), self.load_corpus(test_path)
            self.data = self.folds['all']

        self.assign_labels()

        return self

    def assign_labels(self):
        """ Assign labels """
        if isinstance(self.label, str):
            for fold in self.folds:
                self.folds[fold]['label_str'] = self.label
            self.data['label_str'] = self.label
        else: # is a dict mapping dataset labels to other labels
            for fold in self.folds:
                self.folds[fold]['label_str'] = self.folds[fold].label.map(self.label)
            self.data['label_str'] = self.data.label.map(self.label)

    def filter_lda(self):
        """ Remove posts, or only select posts, that match certain topics in a trained LDA model """
        # Load trained model, vectorizer
        print("Filtering by LDA topics...")
        with open(self.lda_filter['model'], 'rb') as f:
            lda = pickle.load(f)
        with open(self.lda_filter['vectorizer'], 'rb') as f:
            vectorizer = pickle.load(f)
        if self.lda_filter['query'] is None:
            selected = self.folds['all']
        else:
            selected = self.folds['all'].query(self.lda_filter['query']).copy()
            rest = self.folds['all'][~self.folds['all'].index.isin(selected.index)]

        # Infer topics on documents
        bow = vectorizer.transform(selected.text)
        doc_topics = lda.transform(bow)
        assigned_topics = np.argmax(doc_topics, axis=1)
        
        # Filter out documents with certain topics
        selected['topic'] = assigned_topics
        filtered = selected
        if 'include_topics' in self.lda_filter.keys() and self.lda_filter['include_topics'] is not None:
            filtered = filtered[filtered.topic.isin(self.lda_filter['include_topics'])]
        if 'exclude_topics' in self.lda_filter.keys() and self.lda_filter['exclude_topics'] is not None:
            filtered = filtered[~filtered.topic.isin(self.lda_filter['exclude_topics'])]
        if self.lda_filter['query'] is None:
            self.folds['all'] = filtered
        else:
            self.folds['all'] = pd.concat([filtered.drop(columns='topic'), rest]).sort_index()

    def sample(self):
        """ Sample particular portions of the corpus """
        for fold, data in self.folds.items():
            if fold == 'test':
                continue # don't sample test sets: use full, unduplicated ones
            if 'query' in self.sample_info:
                selected = data.query(self.sample_info['query'])
            else:
                selected = data
            rest = data[~data.index.isin(selected.index)]
            frac = self.sample_info.get('sample_factor', None)
            replace = False
            if frac is not None and frac > 1:
               replace = True 
            sampled = selected.sample(n=self.sample_info.get('sample_n', None), 
                    frac=frac, replace=replace, random_state=9)
            self.folds[fold] = pd.concat([sampled, rest]).sort_index()

    def print_save_stats(self):
        """ Print, save out stats on the dataset in a log or output dir 
            (#posts, #words per domain type, overall)
        """
        stats = self.data.groupby('domain').agg({'word_count': ['count', 'sum', 'mean']}).reset_index()
        stats.columns = ['domain', 'post_count', 'word_count', 'avg_post_words']
        total = pd.DataFrame({'domain': 'all', 'post_count': len(self.data), 'word_count': self.data.word_count.sum(),
            'avg_post_words': self.data.word_count.mean()}, index=['total'])
        stats = pd.concat([stats,total])
        stats['post%'] = stats['post_count']/stats.loc['total', 'post_count']
        stats['word%'] = stats['word_count']/stats.loc['total', 'word_count']
        stats = stats[['domain', 'post_count', 'post%', 'word_count', 'word%', 'avg_post_words']]
        print(f"Corpus {self.name} stats:")
        print(stats)
        stats.to_json(self.stats_fpath, orient='records', lines=True)

    def set_labels(self, label2id):
        """ Set label column based on the supplied label2id dict """
        for fold in self.folds:
            self.folds[fold]['label'] = self.folds[fold]['label_str'].map(label2id)
        self.data['label'] = self.data['label_str'].map(label2id)
        return self

    def set_data(self, foldname):
        """ Set the data attribute to the specified fold """
        self.data = self.folds[foldname]
        return self

    def save(self):
        """ Save out corpus data for easier loading """
        print(f"Saving corpus to {self.fpath}...")
        self.data.to_json(self.fpath, orient='table', indent=4)
        print(f"Saving corpus to {self.tmp_fpath}...") # for faster loading as an option
        self.data.to_pickle(self.tmp_fpath)
        if self.split is not None:
            path = Path(self.fpath)
            tmp_fpath = Path(self.tmp_fpath)
            train_json_path = str(path.with_stem(str(path.stem) + self.train_suffix))
            test_json_path = str(path.with_stem(str(path.stem) + self.test_suffix))
            train_pkl_path = str(tmp_fpath.with_stem(str(tmp_fpath.stem) + self.train_suffix))
            test_pkl_path = str(tmp_fpath.with_stem(str(tmp_fpath.stem) + self.test_suffix))
            self.folds['train'].to_json(train_json_path, orient='table', indent=4)
            self.folds['train'].to_pickle(train_pkl_path)
            self.folds['test'].to_json(train_json_path, orient='table', indent=4)
            self.folds['test'].to_pickle(test_pkl_path)

    @classmethod
    def load_corpus(cls, path, split=False, train_suffix=None, test_suffix=None):
        """ Load a corpus from disk and return it as a dataframe"""
        res = None
        if path.endswith('.json'):
            res = pd.read_json(path, orient='table')
        elif path.endswith('.pkl'):
            if split:
                res = {}
                path = Path(path)
                train_pkl_path = str(path.with_stem(str(path.stem) + train_suffix))
                test_pkl_path = str(path.with_stem(str(path.stem) + test_suffix))
                res['train'] = pd.read_pickle(train_pkl_path)
                res['test'] = pd.read_pickle(test_pkl_path)
                res['all'] = pd.read_pickle(path)
            else:
                res = pd.read_pickle(path)
        return res
