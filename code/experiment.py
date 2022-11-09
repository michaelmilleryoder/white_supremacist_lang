""" Train a classifier on a corpus and evaluate it with specified settings """
import pdb

import pandas as pd
import numpy as np

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, name: str, train: bool, test: bool, train_corpora: list[Corpus], test_corpora: list[Corpus], 
        classifier: dict, test_label_combine: dict = None):
        """ Args:
                name: name of the model for saving out results
                train: whether to train a model
                test: whether to evaluate a model
                train_corpora: training corpora to be combined and given labels based on `label_str' column
                test_corpora: evaluation corpora, unseen by the model. Each should already have a column of 'label'
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
                test_label_combine: a dictionary of any changes of predicted labels to make 
                    (to combine 3-way to 2-way classification, eg)
        """
        self.name = name
        self.do_train = train
        self.do_test = test
        self.test_corpora = test_corpora
        self.train_data = pd.concat([train_corpus.data for train_corpus in train_corpora])
        self.clf = None
        self.label2id = None
        self.test_label_combine = test_label_combine
        if classifier['type'] == 'bert':
            train_length = None
            id2label = dict(enumerate(self.train_data['label_str'].astype('category').cat.categories))
            self.label2id = {l: k for k, l in enumerate(self.train_data['label_str'].astype('category').cat.categories)}
            n_labels = len(id2label)
            if self.do_train:
                train_length = sum([len(train_corpus.data) for train_corpus in train_corpora])
                n_labels = len(self.train_data['label_str'].unique())
                self.train_data['label'] = self.train_data['label_str'].astype('category').cat.codes
            self.clf = BertClassifier(self.name, self.do_train, load=classifier['load'], 
                train_length=train_length, 
                n_labels = n_labels,
                id2label = id2label,
                label2id = self.label2id,
                n_epochs=classifier['n_epochs'],
                test_label_combine=self.test_label_combine,
                )
        
    def run(self):
        if self.do_train:
            self.train()
        if self.do_test:
            self.evaluate()
        
    def train(self):
        # Prepare training set, including labels
        #train_data = pd.concat([self.train_pos.data, self.train_neg.data])
        # List of arrays of labels, 0 or 1 for each class (1-hot encoded)
        #self.train_data['label'] = [arr[0] for arr in np.split(pd.get_dummies(self.train_data['label_str']).values, 
        #                                len(self.train_data), axis=0)]
        # Trying just passing indices instead of one-hot arrays (so shape of 1)
        #self.train_data['label_str'] = self.train_data['label_str'].astype('category')

        # Train
        self.clf.train(self.train_data)
    
    def evaluate(self):
        # Apply labels to test corpora (should probably do as a function called in corpus.py)
        for corpus in self.test_corpora:
            corpus.data['label'] = corpus.data['label_str'].map(self.label2id)
        self.clf.evaluate(self.test_corpora)
