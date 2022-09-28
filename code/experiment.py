""" Train a classifier on a corpus and evaluate it with specified settings """

import pandas as pd

from bert_classifier import BertClassifier
from corpus import Corpus


class Experiment:

    def __init__(self, train_pos: Corpus, train_neg: Corpus, test: Corpus, classifier: dict):
        """ Args:
                train_pos: training corpus of positive class (white supremacist) examples
                train_neg: training corpus of negative class (non-white supremacist) examples
                test: evaluation corpus, unseen by the model. Should already have a column of 'label' in test.corpus
                classifier: dict of info on the classifier. Should include a key of 'type' with a string in {bert}
        """
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.test = test
        self.clf = None
        if classifier['type'] == 'bert':
            self.clf = BertClassifier(classifier['load'])
        
    def run(self):
        self.train()
        self.evaluate()
        
    def train(self):
        # Prepare training set
        self.train_pos.data['label'] = 1
        self.train_neg.data['label'] = 0
        train_data = pd.concat([self.train_pos.data, self.train_neg.data])

        # Train
        self.clf.train(train_data)
    
    def evaluate(self):
        self.clf.evaluate(self.test.data)
