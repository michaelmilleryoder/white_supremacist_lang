""" Train a classifier on a corpus and evaluate it with specified settings """
import pdb

import pandas as pd

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, name: str, train: bool, test: bool, train_corpora: list[Corpus], test_corpora: list[Corpus], 
        classifier: dict):
        """ Args:
                name: name of the model for saving out results
                train: whether to train a model
                test: whether to evaluate a model
                train_corpora: training corpora to be combined and given labels based on `label_str' column
                test_corpora: evaluation corpora, unseen by the model. Each should already have a column of 'label'
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
        """
        self.name = name
        self.do_train = train
        self.do_test = test
        self.test_corpora = test_corpora
        self.train_data = pd.concat(train_corpora)
        self.clf = None
        if classifier['type'] == 'bert':
            train_length = None
            n_labels = 3 # TODO: need to consider this for running just evaluation
            if self.do_train:
                train_length = sum([len(train_corpus.data) for train_corpus in train_corpora])
                n_labels = len(self.train_data['label_str'].unique())
            self.clf = BertClassifier(self.name, self.do_train, load=classifier['load'], 
                train_length=train_length, 
                n_labels=n_labels,
                n_epochs=classifier['n_epochs'])
        
    def run(self):
        if self.do_train:
            self.train()
        if self.do_test:
            self.evaluate()
        
    def train(self):
        # Prepare training set, including labels
        #train_data = pd.concat([self.train_pos.data, self.train_neg.data])
        train_data['label'] = pd.get_dummies(train_data['label_str']).values
        pdb.set_trace() # check the format of label (want lists for each cell)

        # Train
        self.clf.train(train_data)
    
    def evaluate(self):
        self.clf.evaluate(self.test_corpora)
