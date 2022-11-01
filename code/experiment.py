""" Train a classifier on a corpus and evaluate it with specified settings """
import pdb

import pandas as pd

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, name: str, train: bool, test: bool, train_pos: Corpus, train_neg: Corpus, 
            test_corpora: list[Corpus], classifier: dict):
        """ Args:
                name: name of the model for saving out results
                train: whether to train a model
                test: whether to evaluate a model
                train_pos: training corpus of positive class (white supremacist) examples
                train_neg: training corpus of negative class (non-white supremacist) examples
                test_corpora: evaluation corpora, unseen by the model. Each should already have a column of 'label'
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
        """
        self.name = name
        self.do_train = train
        self.do_test = test
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.test_corpora = test_corpora
        self.clf = None
        if classifier['type'] == 'bert':
            train_length = None
            if self.do_train:
                train_length = len(self.train_pos.data) + len(self.train_neg.data)
            self.clf = BertClassifier(self.name, self.do_train, load=classifier['load'], train_length=train_length)
        
    def run(self):
        if self.do_train:
            self.train()
        if self.do_test:
            self.evaluate()
        
    def train(self):
        # Prepare training set
        self.train_pos.data['label'] = 1
        self.train_neg.data['label'] = 0
        train_data = pd.concat([self.train_pos.data, self.train_neg.data])

        # Train
        self.clf.train(train_data)
    
    def evaluate(self):
        self.clf.evaluate(self.test_corpora)
