""" Train a classifier on a corpus and evaluate it with specified settings """
import pdb

import pandas as pd

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, name: str, train: bool, test: bool, train_pos: Corpus, train_neg: Corpus, 
            test_corpora: list[Corpus], test_by_dataset: bool, classifier: dict):
        """ Args:
                name: name of the model for saving out results
                train: whether to train a model
                test: whether to evaluate a model
                train_pos: training corpus of positive class (white supremacist) examples
                train_neg: training corpus of negative class (non-white supremacist) examples
                test_corpora: evaluation corpora, unseen by the model. Each should already have a column of 'label'
                test_by_dataset: whether to evaluate each dataset separately in the test corpus
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
        """
        self.name = name
        self.do_train = train
        self.do_test = test
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.test_corpora = test_corpora
        self.test_by_dataset = test_by_dataset
        self.clf = None
        if classifier['type'] == 'bert':
            self.clf = BertClassifier(self.name, self.do_train, load=classifier['load'])
        
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
        for test_corpus in self.test_corpora:
            self.clf.evaluate(test_corpus.data, self.test_by_dataset, name=test_corpus.name)
