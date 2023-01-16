""" Implementation of dictionary-based classifier for white nationalism described in Siegel et al. 2021.
    The classifier first matches words from a dictionary (list) and then runs a Naive Bayes classifier
    to do binary white nationalism classification.

    @author Michael Miller Yoder (of process from Alexandra Siegel and others)
    @year 2023
"""

import os
import pickle
import re
from string import punctuation
import pdb

import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

from corpus import Corpus


class StemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(t.lower()) for t in word_tokenize(doc)]


class Siegel2021DictClassifier:
    """ White nationalism dictionary-based classifier from Siegel et al. 2021 """

    def __init__(self, exp_name: str, load = None):
        """ Args:
                exp_name: name of the experiment (for the output filename)
                load: None to train the model (from its own data), or a path to the model to load
        """
        self.exp_name = exp_name
        words_df = pd.read_csv('../data/siegel2021/qjps_hatespeech_dictionary.csv')
        words_df = words_df[words_df.exclude != 'yes']
        wn_types = ['anti_semitic_white nationalist', 'white nationalist', 'white nationalist ', 'white_nationalist']
        self.wn_words = set(words_df[words_df.type.isin(wn_types)]['term'])
        self.output_dir = f"../output/siegel2021_dict_classifier"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.clf_path = os.path.join(self.output_dir, 'nb_clf.pkl')
        if load:
            with open(self.clf_path, 'rb') as f:
                self.clf = pickle.load(f)
        else: # train the NB classifier
            self.clf = self.train_nb()
        self.id2label = {0: 'neutral', 1: 'white_supremacist'}
        self.label2id = {val: key for key, val in self.id2label.items()}

    def train_nb(self):
        """ Train the Naive Bayes filter classifier from its own data, not any provided to this classifier. 
            Saves the model out. """
        print("Training the Naive Bayes classifier...")
        trainpath = '../data/siegel2021/white_nationalist_training_data.csv'
        train = pd.read_csv(trainpath, index_col=0)
        stops = list(punctuation.replace('@', '').replace('#', ''))
        vec = CountVectorizer(tokenizer=StemTokenizer(), stop_words=stops)
        bow = vec.fit_transform(train.text)
        clf = MultinomialNB()
        clf.fit(bow, train.white_nationalism_total)
        with open(self.clf_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Saved Naive Bayes classifier to {self.clf_path}")
        return clf

    def dict_filter(self, text_col: pd.Series):
        """ Apply the dictionary filter to a pandas Series. 
            Returns True for every row that it matches, False for those that don't. """

        # See which terms are present at all
        vocab = set(sum([text_col.str.split()], []))
        present_dict = self.wn_words.intersection(vocab)
        wn_pattern = re.compile('|'.join(present_dict).encode('unicode_escape')) # assumes input text lowercased
        return text_col.str.contains(wn_pattern)

    def evaluate(self, test: list[Corpus]):
        """ Evaluate the model on unseen corpora. Gives separate evaluations per dataset within the corpora
            Args:
                test: list of Corpus objects with pandas DataFrame of unseen data in corpus.data, 
                    containing 'text' and 'label' columns
        """
        result_lines = []
        for corpus in test:
            print(f"\nEvaluating on test corpus {corpus.name}...")
            for dataset in corpus.data.dataset.unique():
                selected = corpus.data.query('dataset==@dataset').copy()
                
                # Apply the dictionary filter
                contains_terms = self.dict_filter(selected.text)
                pdb.set_trace() # check the ones it finds there
                matches = selected.text[contains_terms]

                # For posts that contain matches, run NB classifier
                preds = self.clf.predict(matches)
                probs = self.clf.predict_proba(matches)
                pred_wn = matches.index[preds.astype(bool)]
                selected['pred_wn'] = 0
                selected.loc[pred_wn, 'pred_wn'] = 1
                pdb.set_trace() # check that it was actually registered predictions

                # Save out class name predictions, probabilities
                class_prob = pd.DataFrame(probs)
                class_prob.columns = class_prob.columns.map(self.id2label)
                prob_outpath = os.path.join(self.output_dir, f'{dataset}_pred_probs.txt')
                class_prob.to_json(prob_outpath, orient='records', lines=True)
                pred_outpath = os.path.join(self.output_dir, f'{dataset}_predictions.json')
                preds_classnames = pd.Series(preds).map(self.id2label).to_json(pred_outpath)

                # Calculate ROC AUC
                score = roc_auc_score(selected.label.map(self.label2id), probs)
                result_lines.append({'dataset': dataset, 'metric': 'roc_auc', 'value': score})

        results = pd.DataFrame(result_lines)
        outpath = os.path.join(self.output_dir, 'results.jsonl')
        results.to_json(outpath, orient='records', lines=True)
        print(f"Saved results to {outpath}")
