""" Train a classifier on a corpus and evaluate it with specified settings """
import pdb

import pandas as pd
import numpy as np

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, name: str, train: bool, test: bool, 
        corpora: dict[str, Corpus], train_corpora: list[dict], test_corpora: list[dict], 
        classifier: dict, test_label_combine: dict = None):
        """ Args:
                name: name of the model for saving out results
                train: whether to train a model
                test: whether to evaluate a model
                corpora: dictionary of name as keys, Corpus objects as values, to be used in the experiment
                train_corpora: list of dictionary of info about which corpora and folds are to be used for training.
                    These are given labels based on `label_str' column
                test_corpora: dictionary of info about which corpora and folds are to be used for evaluation.
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
                test_label_combine: a dictionary of any changes of predicted labels to make 
                    (to combine 3-way to 2-way classification, eg)
        """
        self.name = name
        self.do_train = train
        self.do_test = test
        self.corpora = corpora
        self.train_corpora = train_corpora
        self.train_data = pd.concat([self.corpora[corpus_info['name']].folds[corpus_info.get('fold', 'all')] 
                for corpus_info in self.train_corpora])
        self.remove_train_duplicates()
        self.test_corpora = test_corpora
        self.clf = None
        self.label2id = None
        self.test_label_combine = test_label_combine
        if classifier['type'] == 'bert':
            train_length = None
            id2label = dict(enumerate(self.train_data['label_str'].astype('category').cat.categories))
            self.label2id = {l: k for k, l in enumerate(self.train_data['label_str'].astype('category').cat.categories)}
            n_labels = len(id2label)
            if self.do_train:
                train_length = len(self.train_data)
                n_labels = len(self.train_data['label_str'].unique())
                self.train_data['label'] = self.train_data['label_str'].astype('category').cat.codes
            self.clf = BertClassifier(self.name, self.do_train, load=classifier['load'], 
                    train_length=train_length, 
                    n_labels = n_labels,
                    id2label = id2label,
                    label2id = self.label2id,
                    n_epochs=classifier['n_epochs'],
                    checkpoints = classifier.get('checkpoints', None),
                    test_label_combine=self.test_label_combine,
                )
        
    def run(self):
        if self.do_train:
            self.train()
        if self.do_test:
            self.evaluate()
        
    def train(self):
        # Train
        self.clf.train(self.train_data)
    
    def evaluate(self):
        # Apply labels to test corpora (should probably do as a function called in corpus.py)
        test_corpora = []
        for corpus_info in self.test_corpora:
            corpus = self.corpora[corpus_info['name']]
            corpus.set_labels(self.label2id)
            corpus.set_data(corpus_info.get('fold', 'all'))
            test_corpora.append(corpus)
        self.clf.evaluate(test_corpora)

    def remove_train_duplicates(self):
        """ Remove training set duplicates that may emerge from combining corpora """
        # Sort so that instances with labels are first
        if 'label' in self.train_data.columns:
            self.train_data.sort_values(['label'], inplace=True)
        elif 'label_str' in self.train_data.columns:
            self.train_data.sort_values(['label_str'], inplace=True)
        self.train_data.drop_duplicates('text', inplace=True)
