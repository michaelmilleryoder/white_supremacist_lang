""" Storing and processing datasets, including processed versions 
    This module includes a superclass and subclasses for loading specific datasets
"""

import os
import json
import re
import pdb
from multiprocessing import Pool
import datetime
import itertools

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
import spacy

import utils
from utils import (remove_mentions, remove_urls, tokenize_lowercase, process_4chan,
        process_tweet, process_tweet_text, process_article, process_reddit, process_chat, 
        load_now, process_now, process_rieger2021)


def corpus_year_count(corpus):
    """ Returns a table of post and word counts per year in a corpus (often for matching samples) 
        corpus: pandas DataFrame of data used as a corpus. Must have a timestamp datetime column
    """
    #corpus['word_count'] = corpus.text.str.split().str.len()
    yearly = corpus.groupby(by=corpus.timestamp.dt.year).agg({
        'text': 'count',
        'word_count': ['sum', 'mean'],
        })
    lookup = pd.DataFrame(yearly)
    #lookup['begin'] = pd.to_datetime(yearly.index.astype(int).astype(str), format='%Y')
    #lookup['end'] = [x.replace(year=x.year + 1) for x in lookup['begin']]
    lookup.index.name = 'year'
    lookup.index = lookup.index.astype(int)
    lookup.rename(columns={'text': 'post_count'}, inplace=True)
    return lookup


class Dataset:
    """ Superclass for storing data and attributes of datasets """

    @classmethod
    def all_subclasses(cls):
        """ Return all subclasses of the class, as deep as they go (recursive) """
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in c.all_subclasses()])

    def __new__(cls, name, source, domain, load_paths, ref_corpora=None, match_factor=1, min_word_limit=1,
            include_users: bool = False):
        """ Choose the right subclass based on dataset name """
        subclass_name = f"{name.capitalize()}Dataset"
        #subclass_map = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
        subclass_map = {subclass.__name__: subclass for subclass in cls.all_subclasses()}
        #subclass = subclass_map.get(subclass_name.split('_')[0], cls)
        subclass = subclass_map.get(subclass_name, cls)
        instance = super(Dataset, subclass).__new__(subclass)
        return instance

    def __init__(self, name, source, domain, load_paths, ref_corpora=None, match_factor=1, min_word_limit=1,
            include_users: bool = False):
        """ Args:
                name: dataset name
                source: source platform
                domain: discourse type of the source (tweet, chat, forum, long-form (article or book))
                load_paths: list of arguments for loading the datasets (like file paths)
                ref_corpora: a dict with keys as corpora names and values dataframes of other corpora 
                    to use in assembling this dataset
                match_factor: factor to multiply even sample with reference corpus by
                min_word_limit: Minimum word limit of posts. Put 1 to take all posts
                include_users: whether to include a column of usernames (if available)
        """
        self.name = name
        self.source = source
        self.domain = domain
        #self.loader = getattr(load_process_dataset, f'{self.name.capitalize()}Loader')
        self.load_paths = load_paths
        self.ref_corpora = {}
        if ref_corpora is not None:
            self.lookup = {}
            for name, corpus in ref_corpora.items():
                self.ref_corpora[name] = corpus.query(f'domain=="{domain}"').copy()
                self.lookup[name] = corpus_year_count(self.ref_corpora[name])
        self.match_factor = match_factor
        self.min_word_limit = min_word_limit
        self.include_users = include_users
        self.n_jobs = 20 # number of processes for multiprocessing of data
        self.data = pd.DataFrame()
        #cls.get_subclass()

    def uniform_format(self, timestamp_col=None, unit=None, format=None, errors='raise'):
        """ Format the dataframe for combining with other datasets
            Set a column to a timestamp data type if there is one
            Create an index for the dataset with the dataset name.
            Add a column of the dataset name, source and domain.
            Filter to posts above a word_limit (shorter often didn't reveal the ideology).
            Filter self.data to just the columns needed from all datasets:
                text, word_count, timestamp (if present), dataset, source, domain
            Args:
                timestamp_col: the name of a column to convert to a datetime data type. If None, no timestamp
                unit: Additional parameter to pass to pd.to_datetime
                format: Additional parameter to pass to pd.to_datetime
                errors: Additional parameter to pass to pd.to_datetime
        """

        # Set index first so regardless of filtering will have the same
        self.data['id'] = self.name + '_' + self.data.index.astype(str)
        self.data.set_index('id', inplace=True)
        assert self.data.index.duplicated(keep=False).any() == False # Any duplicates indices
        #self.data = self.data[self.data['text'].str.split().str.len() >= min_word_limit]
        #with Pool(self.n_jobs) as p:
        #    self.data['word_count'] = list(tqdm(p.imap(word_count, self.data.text), total=len(self.data), ncols=60))
        self.data = self.data[self.data['word_count'] >= self.min_word_limit]
        self.data.drop_duplicates(subset='text', keep='first', inplace=True)
        if timestamp_col is not None:
            self.data['timestamp'] = pd.to_datetime(self.data[timestamp_col], format=format, utc=True, unit=unit, errors=errors)
        self.data['dataset'] = self.name
        self.data['source'] = self.source
        self.data['domain'] = self.domain
        selected_cols = ['text', 'word_count', 'dataset', 'source', 'domain'] + \
                [col for col in ['timestamp', 'label', 'user'] if col in self.data.columns]
        self.data = self.data[selected_cols]

    def load(self):
        """ Usually overridden in a subclass """
        self.data = pd.read_csv(self.load_paths[0])

    def process(self):
        """ Needs to be overridden in a subclass """
        pass

    def print_stats(self):
        """ Print statistics on number of posts and word count per year compared to the reference corpus
        """

        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]

        # Comparison to matching white supremacist data
        print('\t\tMatching white supremacist corpus #posts: '
                f'{len(self.ref_corpora[ws_match])}')
        print(f'\t\t{self.name} #posts: {len(self.data)}')
        print('\t\tMatching white supremacist corpus word count mean: '
                f'{self.ref_corpora[ws_match].word_count.mean()}')
        print(f'\t\t{self.name} word count mean: {self.data.word_count.mean()}')
        print('\t\tMatching white supremacist corpus word count sum: '
                f'{self.ref_corpora[ws_match].word_count.sum()}')
        print(f'\t\t{self.name} word count sum: {self.data.word_count.sum()}')


class RawTwitter(Dataset):
    """ Class for handling scraped Twitter JSON """

    def process(self):
        """ Process Twitter data for combining with other datasets.
        """
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['processed_text'], self.data['word_count'] = list(zip(*[process_tweet(
            text, user_mentions, urls, tokenizer) for text, user_mentions, urls in tqdm(zip(
            self.data['text'], self.data['entities.mentions'], self.data['entities.urls']), 
            total=len(self.data), ncols=60)]))
        self.data = self.data[~self.data.processed_text.str.contains(
            "account is temporarily unavailable because it violates the twitter media policy")]
        self.data.drop(columns='text', inplace=True)
        self.data.rename(columns={'id': 'tweet_id', 'processed_text': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='created_at')


class Qian2018Dataset(RawTwitter):

    def load(self):
        """ Load tweets. Rehydrated based on tweet IDs of particular ideologies in ws_data_datasets.ipynb """
        with open(self.load_paths[0], 'r') as f:
            self.data = pd.json_normalize([json.loads(tweet) for tweet in f.read().splitlines()])


class Elsherief2021Dataset(RawTwitter):

    def load(self):
        """ Load tweets """
        elsherief2021_tweets_path, elsherief2021_users_path, qian2018_users_path, elsherief2021_stg2_path = self.load_paths

        # Load rehydrated ElSherief+2021 tweet data
        with open(elsherief2021_tweets_path, 'r') as f:
            tweet_data = pd.json_normalize([json.loads(tweet) for tweet in f.read().splitlines()])
            tweet_data.id = tweet_data.id.astype('int64')

        # Load ElSherief+2021 users
        with open(elsherief2021_users_path, 'r') as f:
            user_data = pd.json_normalize([json.loads(user) for user in f.read().splitlines()])

        # Merge in user info
        user_data.drop_duplicates(subset='id', inplace=True)
        user_data.set_index('id', drop=True, inplace=True)
        elsherief2021_hydrated = tweet_data.join(user_data, on='author_id', rsuffix='_user')

        # Use Qian+2018 labels to filter tweets based on ideology
        with open(qian2018_users_path, 'r') as f:
            qian2018_user_data = pd.json_normalize([json.loads(user) for user in f.read().splitlines()])
        qian2018_user_data.drop_duplicates(subset='id', inplace=True)
        qian2018_user_data.set_index('id', drop=True, inplace=True)
        user_matches = elsherief2021_hydrated[elsherief2021_hydrated['author_id'].isin(qian2018_user_data.index)]

        # White grievance tweets
        # Load stage 2 annotations, implicit categories which include white grievance
        stg2 = pd.read_csv(elsherief2021_stg2_path, sep='\t')
        white_grievance = stg2.query('implicit_class=="white_grievance" or extra_implicit_class=="white_grievance"').rename(columns={'post': 'text'})

        # Add white grievance to hydrated and filtered ElSherief+2021 tweets
        self.data = pd.concat([white_grievance, user_matches]).drop_duplicates(subset='id').reset_index(drop=True)
        self.data.rename(columns={'id': 'tweet_id'}, inplace=True)
        

class PatriotfrontDataset(Dataset):

    def load(self):
        """ Load data dumps """
        dirpath2017, dirpath2018_fac, dirpath2018_mgs, names_fpath = self.load_paths
        message_dfs = []
        for dirpath in [dirpath2017, dirpath2018_fac, dirpath2018_mgs]:
            channels = pd.read_csv(os.path.join(dirpath, 'channels.csv'), index_col=0)
            messages = pd.read_csv(os.path.join(dirpath, 'messages.csv'))
            messages.dropna(subset='message', inplace=True)
            messages = messages.join(channels, on='channel_id', rsuffix='_channel')
            messages = messages.query('name == "general"')
            message_dfs.append(messages) 
        self.data = pd.concat(message_dfs).reset_index()

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Add user info
        if self.include_users:
            self.data['user'] = self.data.user_id 

        # Remove any common first names
        names = pd.read_csv(self.load_paths[-1], skiprows=[0])
        names = names.query('Rank <= 300')
        common_names = set(names['Name'].str.lower()).union(names['Name.1'].str.lower())
        # Remove spencer, guy (for Richard Spencer)
        common_names -= {'spencer', 'guy'}

        tokenizer = TweetTokenizer(strip_handles=True)
        #zipped = zip(self.data['message'], itertools.repeat(tokenizer), itertools.repeat(common_names))
        self.data['text'], self.data['word_count'] = list(zip(*[process_chat(
            m, tokenizer, common_names) for m in self.data['message']]))
        #self.data['text'], self.data['word_count'] = list(zip(*[tokenize_remove(
        #    text, common_names) for text in self.data['message']]))
        self.uniform_format(timestamp_col='timestamp')


class IronmarchDataset(Dataset):

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Add user info
        if self.include_users:
            self.data['user'] = self.data.index_author 

        # Tokenize, etc
        with Pool(self.n_jobs) as p:
            self.data['processed'], self.data['word_count'] = list(zip(*tqdm(p.imap(
                    tokenize_lowercase, self.data['index_content']), total=len(self.data), ncols=60)))
        self.data.reset_index(drop=True, inplace=True)
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='index_date_created', unit='s')


class StormfrontDataset(Dataset):

    @classmethod
    def preprocess(cls, inp, nlp):
        text = re.sub(r'Quote:\n\n\n\n\nOriginally Posted by .*\n\n\n', '', inp) # Remove quote tag
        text = re.sub(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', '', text) # Remove URLs
        return tokenize_lowercase(text, nlp)

    def load(self):
        """ Load data dump """
        dfs = []
        for fname in os.listdir(self.load_paths[0]):
            fpath = os.path.join(self.load_paths[0], fname)
            dfs.append(pd.read_csv(fpath))
        self.data = pd.concat(dfs).reset_index(drop=True) 

        # Split up breadcrumb
        self.data[[f'breadcrumb{i}' for i in range(5)]] = self.data.thread_breadcrumb.str.split(' > ', expand=True)

        # Try to remove non-English, non-ideological threads
        exclude = ['Nederland & Vlaanderen', 
                    'Srbija',
                    'en Español y Portugués',
                    'Italia',
                    'Croatia',
                    'South Africa', # some Boer/Dutch
                    'en Français',
                    'Russia',
                    'Baltic / Scandinavia', # but contains lots of English
                    'Hungary', # but contains lots of English
                    'Opposing Views Forum',
                   'Computer Talks'
                   ]

        formatted = [f'Stormfront {el}' for el in exclude]
        self.data = self.data.query('breadcrumb2!=@formatted').dropna(subset='text')

    def process(self):
        """ Process data into a format to combine with other datasets """
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        zipped = zip(self.data.text, itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            #self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(self.preprocess, self.data['text']), 
            #    total=len(self.data), ncols=60)))
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(self.preprocess, tqdm(zipped, 
                total=len(self.data), ncols=60))))
        self.data.reset_index(drop=True, inplace=True)
        self.uniform_format(timestamp_col='timestamp', errors='coerce')


class Jokubauskaite2020Dataset(Dataset):

    def load(self):
        """ Load dataset, select certain threads """
        selected = [
            # 'president trump', # too focused just on Trump
            # 'trump', # too focused just on Trump
            'kraut/pol/ and afd',
            'national socialism',
            # 'islam', # Super Islamophobic and antisemitic but not necessarily white supremacist ideology
            'fascism',
            'dixie',
            # 'hinduism', # Not super white supremacist, though some antisemitism
            # 'black nationalism', # super racist and white nationalist, but some actual Black nationalism
            'kraut/pol/', # yep, German nationalists. Some German, but lots of white supremacy
            'ethnostate',
            'white',
            'chimpout',
            'feminist apocalypse',
            '(((krautgate)))',
        ]

        dfs = []
        for general in selected:
            fpath = os.path.join(self.load_paths[0], f'{general.replace("/", " ")} general.csv')
            dfs.append(pd.read_csv(fpath, low_memory=False))

        self.data = pd.concat(dfs).reset_index(drop=True)

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Add user info
        if self.include_users:
            self.data['user'] = self.data['author'] # username, if provided (rarely is)
            self.data['user'] = self.data.user.replace('Anonymous', np.nan)

        # Tokenize, etc
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        zipped = zip(self.data.body, itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            #self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(process_4chan, self.data.body), 
            #    total=len(self.data), ncols=60)))
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(process_4chan, tqdm(zipped, 
                    total=len(self.data), ncols=60))))
        self.uniform_format(timestamp_col='timestamp')


class Papasavva2020Dataset(Dataset):

    def load(self):
        """ Load dataset, created in ws_data_datasets.ipynb """
        self.data = pd.read_csv(self.load_paths[0],index_col=0, low_memory=False).reset_index(drop=True)

    def process(self):
        """ Process data into a format to combine with other datasets """
        self.data = self.data.drop(columns='id').rename(columns={'no': 'id', 'com': 'body'})
        # Remove duplicates with jokubausaite2020
        # Don't like loading it again, but easiest way for now since no way to access other unprocessed dataset
        selected = [
            'kraut/pol/ and afd',
            'national socialism',
            'fascism',
            'dixie',
            'kraut/pol/', # yep, German nationalists. Some German, but lots of white supremacy
            'ethnostate',
            'white',
            'chimpout',
            'feminist apocalypse',
            '(((krautgate)))',
        ]
        dfs = []
        for general in selected:
            fpath = os.path.join(self.load_paths[1], f'{general.replace("/", " ")} general.csv')
            dfs.append(pd.read_csv(fpath, low_memory=False))
        jokubausaite2020_ids = pd.concat(dfs).reset_index(drop=True)['id']
        self.data = self.data[~self.data.id.isin(jokubausaite2020_ids)]

        # Add user info
        if self.include_users:
            self.data['user'] = self.data.trip # user tripcode

        # Tokenize, etc
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'ner'])
        zipped = zip(self.data.body, itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            #self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(process_4chan, self.data.body), 
            #        total=len(self.data), ncols=60)))
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(process_4chan, tqdm(zipped, 
                    total=len(self.data), ncols=60))))
        self.uniform_format(timestamp_col='time', unit='s')


class Calderon2021Dataset(Dataset):

    def load(self):
        """ Load data dump """
        with open(self.load_paths[0]) as f:
            self.data = pd.json_normalize(json.load(f))
        self.data.index = self.source + '_' + self.data.index.astype(str)

    def process(self):
        """ Process data into a format to combine with other datasets """
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        zipped = zip(self.data['title'] + ' ' + self.data['author_wording'], itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            #self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(
            #        process_article, self.data['title'] + ' ' + self.data['author_wording']), total=len(self.data), ncols=60)))
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(process_article, 
                tqdm(zipped, total=len(self.data), ncols=60))))
        # remove date errors. Could extract real date by parsing text
        self.data.loc[~self.data.date.str.startswith('20'), 'date'] = '' 
        self.data = self.data.drop(columns=['author_wording', 'title'])
        self.uniform_format(timestamp_col='date', errors='coerce')


class Pruden2022Dataset(Dataset):
    source_years = {
            'breivik_manifesto': 2011,
            'powell_rivers_of_blood_speech': 1968,
            'raspail_camp_of_the_saints_book': 1973,
            'lane_white_genocide_manifesto': 1988,
            'camus_the_great_replacement_book': 2012,
            'pierce_the_turner_diaries_book': 1978,
            }

    def load(self):
        """ Load document """
        with open(self.load_paths[0], encoding='latin-1') as f:
            if self.source == 'breivik_manifesto':
                text = [line.strip() for line in re.split(r'\n\s+', f.read()) if len(line.strip()) > 0]
            else:
                text = [line.strip() for line in f.read().splitlines() if len(line.strip()) > 0]
        self.data = pd.DataFrame({'orig_text': text, 'year': self.source_years[self.source]})
        self.data.index = self.source + '_' + self.data.index.astype(str)

    def process(self): 
        """ Process data into a format to combine with other datasets """
        self.data['text'], self.data['word_count'] = list(zip(*map(tokenize_lowercase, tqdm(self.data['orig_text'], total=len(self.data), ncols=60))))
        self.uniform_format(timestamp_col='year', format='%Y')


class RawReddit(Dataset):
    """ Parent class for loading and processing data scraped from Reddit through PushShift. """

    def load(self):
        """ Load prescraped Reddit data (from get_reddit.py) """
        fpaths = sorted([fname for fname in os.listdir(self.load_paths[0]) if fname.endswith('.json')])
        dfs = []
        for fname in tqdm(fpaths, ncols=60):
            # print(fname)
            fpath = os.path.join(self.load_paths[0], fname)
            sub = pd.read_json(fpath, orient='table')
            subreddit = fname.split('_')[0]
            dfs.append(sub.assign(subreddit=subreddit.lower()))
        self.data = pd.concat(dfs).reset_index(drop=True)


class Reddit_matchDataset(RawReddit):
    """ Neutral (non-white supremacist) Reddit data that matches forum data in white supremacy dataset """

    def process(self):
        """ Sample data to match forum data in white supremacist data by year (random across subreddits).
            Process data for combining with other datasets in neutral corpus.
        """
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        self.data['timestamp'] = pd.to_datetime(self.data['created_utc'], unit='s', utc=True)
        self.data.reset_index(drop=True, inplace=True)
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        self.data = self.data.groupby(self.data.timestamp.dt.year).apply(lambda group: group.sample(
            int(self.lookup[ws_match][('post_count', 'count')][group.name] * 2 * self.match_factor),
            random_state=9)).reset_index(drop=True)
                # double sample to meet white supremacist forum data length
        zipped = zip(self.data.body, itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            #self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(
            #        process_reddit, self.data['body']), total=len(self.data), ncols=60)))
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(process_reddit, 
                tqdm(zipped, total=len(self.data), ncols=60))))
        self.uniform_format()


class Reddit_antiracistDataset(RawReddit):
    """ Antiracist Reddit data, filled in with neutral data to match the volume of forum data in white supremacy dataset """

    def process(self):
        """ Sample data to match forum data in white supremacist data by year (random across subreddits).
            Process data for combining with other datasets in antiracist corpus.
        """
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        zipped = zip(self.data.body, itertools.repeat(nlp))
        with Pool(self.n_jobs) as p:
            self.data['text'], self.data['word_count'] = list(zip(*p.starmap(process_reddit, 
                tqdm(zipped, total=len(self.data), ncols=60))))
        self.uniform_format(timestamp_col='created_utc')


class Discord_matchDataset(Dataset):
    """ Neutral (non-white supremacist) Discord data to match chat data in white supremacy dataset """
    
    def load(self):
        """ Load Discord random dataset """
        fpaths = [os.path.join(self.load_paths[0], fname) for fname in os.listdir(self.load_paths[0]) if fname.endswith('.txt')]
        dfs = []
        for fpath in tqdm(fpaths, ncols=60):
            with open(fpath) as f:
                dfs.append(pd.DataFrame({'message': [message for line in f.read().splitlines() for message in line.split('\t')]}))
        self.data = pd.concat(dfs)

    def process(self):
        """ Sample data to match chat data in white supremacist training dataset.
            Process data for combining with other datasets in neutral corpus.
        """
        # Compute the avg words per post
        tokenizer = TweetTokenizer(strip_handles=True)

        # Iteratively sample and process, until match the word count of white supremacist matching chat data
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        ws_word_count = int(self.ref_corpora[ws_match].word_count.sum() * self.match_factor)
        ws_len = len(self.ref_corpora[ws_match])
        sampled_words = 0
        sampled = []
        pbar = tqdm(total=int(ws_word_count*1.3), ncols=60) # sometimes progress bar goes over
        while sampled_words < ws_word_count:
            # Sample, tokenize and process
            sample = self.data.sample(ws_len*2) # generally has fewer words/post. Might lead to duplicates, but drop them later
            zipped = zip(sample['message'], itertools.repeat(tokenizer))
            with Pool(self.n_jobs) as p:
                sample['text'], sample['word_count'] = list(zip(*p.starmap(
                        process_chat, zipped)))
                filtered = sample[sample.word_count >= self.min_word_limit]
            sample_words = filtered.word_count.sum()
            sampled_words += sample_words
            sampled.append(filtered)
            pbar.update(sample_words)
        self.data = pd.concat(sampled).drop_duplicates().reset_index()
        self.uniform_format()


class News_matchDataset(Dataset):
    """ Neutral (non-white supremacist) news data to match long-form article data in white supremacy dataset """

    def load(self):
        """ Load NOW news corpus """
        countries = [ # since most articles are from the Daily Stormer and American Renaissance, US-based websites
            'us'
        ]
        fpaths = sorted([os.path.join(self.load_paths[0], fname) for fname in os.listdir(self.load_paths[0]) if re.search(
                r'us', fname, flags=re.IGNORECASE)])

        with Pool(self.n_jobs) as p:
            dfs = list(tqdm(p.imap(load_now, fpaths), total=len(fpaths), ncols=60))
        self.data = pd.concat(dfs)

    def process(self):
        """ Sample data to match long-form data in white supremacist training dataset.
            Process data for combining with other datasets in neutral corpus.
        """
        # Sample specific number of articles by year
        # don't bother about word count since it will be similar once it's truncated with BERT to 512 tokens
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        match = self.data[self.data.year.isin(self.lookup[ws_match].index)]
        self.data = match
        self.data = match.groupby(match.year).apply(lambda group: group.sample(
                int(self.lookup[ws_match].loc[group.name, ('post_count', 'count')]*self.match_factor),
                random_state=9)).reset_index(drop = True)

        # Process data
        with Pool(self.n_jobs) as p:
            self.data['text'], self.data['word_count'] = list(zip(*tqdm(p.imap(
                    process_now, self.data['article']), total=len(self.data), ncols=60)))
        self.uniform_format(timestamp_col='year', format='%Y')


class Twitter_matchDataset(RawTwitter):
    """ Neutral (non-white supremacist) Twitter data to match Twitter data in white supremacy dataset
        Superclass for tweet JSON objects scraped """

    def load(self):
        """ Load tweets collected through keywords at get_tweets_by_query.py (I think) and get_tweets_by_query.ipynb """
        dfs = []
        for fname in tqdm(sorted(os.listdir(self.load_paths[0])), ncols=60):
            with open(os.path.join(self.load_paths[0], fname)) as f:
                dfs.append(pd.json_normalize([json.loads(line) for line in f.read().splitlines()]))
        self.data = pd.concat(dfs).reset_index(drop=True)

    def process(self):
        """ Sample to match current white supremacist corpus distribution of tweets (was initially scraped to
            match an older white supremacist corpus """
        self.data.reset_index(drop=True, inplace=True)
        self.data['created_at'] = pd.to_datetime(self.data['created_at'])
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        self.data = self.data.groupby(self.data.created_at.dt.year).apply(lambda group: group.sample(
            min(int(self.lookup[ws_match][('post_count', 'count')][group.name] * self.match_factor), len(group)),
            random_state=9
            )).reset_index(drop=True)
        super().process()


class Twitter_antiracistDataset(Twitter_matchDataset):
    """ Load and process tweets from antiracist accounts to match Twitter data in white supremacy dataset """

    def process(self):
        """ Sample based on white supremacist yearly counts
            Process data for combining with other datasets in neutral corpus.
        """
        self.data['created_at'] = pd.to_datetime(self.data['created_at'])
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        self.data = self.data[self.data.created_at.dt.year.isin(self.lookup[ws_match].index)]
        self.data = self.data.groupby(self.data.created_at.dt.year).apply(lambda group: group.sample(
                self.lookup[ws_match][('post_count', 'count')][group.name], random_state=9)
                ).reset_index(drop=True)
        super().process()


class Alatawi2021Dataset(Dataset):
    """ Tweets annotated for white supremacy from Alatawi+ 2021 paper """
    
    def process(self):
        """ Process data for evaluating classifiers based on other datasets. """
        self.data['text'], self.data['word_count'] = list(zip(*self.data['input.text'].map(tokenize_lowercase)))
        self.data['label'] = self.data['Voting and Final Labels']
        self.uniform_format()


class Alatawi2021_white_supremacistDataset(Alatawi2021Dataset):
    """ Tweets annotated for white supremacy from Alatawi+ 2021 paper 
        This dataset only keeps the tweets labeled for white supremacy
    """
    
    def load(self):
        """ Load, select only tweets labeled for white supremacy """
        self.data = pd.read_csv(self.load_paths[0])
        self.data = self.data[self.data['Voting and Final Labels']==1]

    def process(self):
        """ Remove label column """
        super().process()
        self.data.drop(columns='label', inplace=True)


class Siegel2021Dataset(Dataset):
    """ Tweets annotated for white nationalism from Siegel+ 2021 paper.
        White nationalist tweets are paired with negative examples marked as not 
        containing white nationalism or hate speech 
    """

    def load(self):
        # Load tweets labeled for white nationalism and for hate speech
        wn = pd.read_csv(self.load_paths[0], index_col=0)
        hs = pd.read_csv(self.load_paths[1])

        # Remove duplicates of both text and annotation cols
        filtered_wn = wn.drop_duplicates(keep='first').copy()
        # Remaining duplicates are ones with disagreeing annotations. Assign them 'yes' since at least one was annotated as such
        filtered_wn.loc[filtered_wn.duplicated(['text'], keep=False), 'white_nationalism_total'] = 'yes'
        filtered_wn.drop_duplicates(keep='first', inplace=True)

        # Remove all duplicates from hate speech (including disagreements, since will be sampling this data anyway)
        hs_nodups = hs.drop_duplicates(keep=False)
        # Remove any white nationalist examples from hate speech set
        hs_nodups = hs_nodups[~hs_nodups.text.isin(wn.text)]

        # Pair white supremacist tweets with non-hate speech
        n_wn = len(filtered_wn.query('white_nationalism_total == "yes"'))
        self.data = pd.concat([
            filtered_wn.query('white_nationalism_total == "yes"').rename(columns={'white_nationalism_total': 'white_nationalism'}),
            #hs_nodups.query('hatespeech == "no"').rename(columns={'hatespeech': 'white_nationalism'}).sample(round(n_wn*7/3)),
            hs_nodups.query('hatespeech == "no"').rename(columns={'hatespeech': 'white_nationalism'}),
        ]).sample(frac=1).reset_index()

    def process(self):
        """ Process data for evaluating classifiers based on other datasets. """
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['text'], self.data['word_count'] = list(zip(*[utils.process_tweet_text(text, 
                tokenizer) for text in self.data['text']]))
        self.data['label'] = self.data['white_nationalism'].map(lambda x: 1 if x=='yes' else 0)
        self.uniform_format()


class Siegel2021_white_nationalist_onlyDataset(Siegel2021Dataset):
    """ Tweets annotated for white nationalism from Siegel+ 2021 paper.
        Only keep white nationalist tweets that are not marked hate speech as positive examples
    """

    def load(self):
        super().load()

        # Load tweets labeled for hate speech 
        hs = pd.read_csv(self.load_paths[1])
        
        # Get hate speech annotations for white nationalist examples
        hs_matches = hs[hs.text.isin(self.data.query('white_nationalism == "yes"').text)]
        match_vals = hs_matches.groupby('text').agg({'hatespeech': lambda x: x[x=='yes'].count()/len(x)})
        hs_examples = match_vals.query('hatespeech >= 0.5')
        
        # Remove white nationalist examples that are marked hate speech
        self.data = self.data[~self.data.text.isin(hs_examples.index)]


class Siegel2021_white_supremacistDataset(Siegel2021Dataset):
    """ (Only) tweets annotated for white nationalism from Siegel+ 2021 paper.
    """

    def load(self):
        super().load()
        self.data = self.data.query('white_nationalism == "yes"')

    def process(self):
        """ Remove label column """
        super().process()
        self.data.drop(columns='label', inplace=True)


class Adl_heatmapDataset(Dataset):
    """ Offline propaganda from ADL HEATMap dataset """

    def load(self):
        # Load annotated unique quotes
        self.data = pd.read_csv(self.load_paths[0])

        # Load quotes from white supremacist groups (already extracted from event descriptions)
        #self.data = pd.read_json(self.load_paths[0], orient='table').reset_index(drop=True)

        #quotes = pd.read_csv(self.load_paths[0])
        #quotes['timestamp'] = pd.to_datetime(quotes.date, format='%m/%d/%y', errors='coerce', utc=True).fillna(
        #        pd.to_datetime(quotes.date, format='%y-%b', errors='coerce', utc=True))

        ## Filter to quotes from groups with white supremacist ideologies
        #incidents = pd.read_csv(self.load_paths[1])
        #incidents['timestamp'] = pd.to_datetime(incidents.date, utc=True)

        #merged = pd.merge(quotes, incidents, how='left', on=['description', 'city', 'timestamp'])
        #self.data = merged[merged['ideology']== 'Right Wing (White Supremacist)']
        #
        ## Remove duplicates
        #self.data = self.data.drop_duplicates(subset='quote').reset_index(drop=True)

    def process(self):
        self.data['text'], self.data['word_count'] = list(zip(*self.data['quote'].str.slice(1,-1).map(tokenize_lowercase)))
        self.data['length'] = self.data.text.str.len()
        self.data = self.data[self.data.length>1]
        self.data = self.data[self.data.text != "\n "]
        self.data['label'] = self.data['Michael'].fillna(self.data['Annotation (propaganda or Not)'])
        self.uniform_format()


class Adl_heatmap_white_supremacistDataset(Adl_heatmapDataset):
    """ Offline propaganda from ADL HEATMap dataset, only keeping those annotated for white supremacist ideology """

    def process(self):
        self.data['text'], self.data['word_count'] = list(zip(*self.data['quote'].str.slice(1,-1).map(tokenize_lowercase)))
        self.data['length'] = self.data.text.str.len()
        self.data = self.data[self.data.length>1]
        self.data = self.data[self.data.text != "\n "]
        self.data['label'] = self.data['Michael'].fillna(self.data['Annotation (propaganda or Not)'])
        # Filter to just data labeled for white supremacist ideology
        self.data = self.data[self.data.label==1]
        self.uniform_format()


class Rieger2021Dataset(Dataset):
    """ Annotated data from Rieger+ 2021 paper """

    @classmethod
    def preprocess(cls, text, nlp):
        """ Preprocess Rieger+ 2021 4chan, 8chan, t_D data """
        # Remove special characters
        text = utils.remove_special(str(text))
        # Tokenize
        return utils.tokenize_lowercase(text, nlp)

    def load(self):
        # Load annotations
        annotations = pd.read_csv(self.load_paths[0], na_values=-99)

        # Rename annotations to English (from codebook)
        en_cols = ['id', 'src', 'spam', 'pers_ins', 'pers_ins_tar1', 'pers_ins_ref1', 'pers_ins_tar2', 'pers_ins_ref2', 'gen_ins',
                    'gen_ins_tar1', 'gen_ins_tar2', 
                    'viol', 'viol_tar', 'stereo', 'stereo2', 'disinfo', 'disinfo_ref', 'ingroup', 'ih_ide'] + \
                    annotations.columns.tolist()[19:]
        annotations.columns = en_cols

        # Load text
        texts = pd.read_excel(self.load_paths[1], sheet_name='Tabelle5')
        texts.drop(columns=[col for col in texts.columns if col.startswith('Unnamed')], inplace=True)

        # Merge in text
        data = pd.merge(texts, annotations, left_on='CommentID', right_on='id')

        data = data.set_index('CommentID').drop(columns=['id', 'src', 'filter_$'])

        data['inhuman_ideology'] = data.ih_ide.astype('category').cat.rename_categories(
            {0: 'none discernible', 1: 'National Socialist', 2: 'white supremacy/white ethnostate'})
        data['inhuman_ideology'].value_counts()

        # Replace numeric codes with names
        demo_categories = {
            1: 'ethnicity',
            2: 'religion',
            3: 'country_of_origin',
            4: 'gender',
            5: 'political_views',
            6: 'sexual_orientation',
            7: 'disability',
            8: 'gender_identity',
            9: 'other',
            -9: 'undetermined'
        }
        identities = {
            1: 'black people',
            2: 'muslims',
            3: 'jews',
            4: 'lgbtq',
            5: 'migrants',
            6: 'people_with_disabilities',
            7: 'social_elites_media',
            8: 'political_opponents',
            9: 'latin_americans',
            10: 'women',
            11: 'criminals',
            12: 'asians',
            13: 'other',
            -9: 'undetermined',
        }

        ref_cols = ['pers_ins_ref1',
                   'pers_ins_ref2']
        tar_cols = ['gen_ins_tar1',
                    'gen_ins_tar2',
                    'viol_tar',
                    'disinfo_ref',
                   ]

        for col in ref_cols:
            data[col] = data[col].astype('category').cat.rename_categories(demo_categories)
        for col in tar_cols:
            data[col] = data[col].astype('category').cat.rename_categories(identities)

        data['Source'] = data.Source.astype('category').cat.rename_categories({
            1: 'reddit', #the_Donald
            2: '4chan', #4chan_pol
            3: '8chan', #8chan_pol
        })

        data['white_supremacist'] = data['inhuman_ideology'].isin(['white supremacy/white ethnostate', 'National Socialist'])

        # Select positive and negative examples
        self.data = data.query('white_supremacist or (gen_ins==0 and viol==0 and pers_ins==0 and not white_supremacist)')

    def process(self):
        nlp = spacy.load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        # Remove NaNs of Text
        self.data = self.data.dropna(subset='Text')
        zipped = zip(self.data.Text, itertools.repeat(nlp))
        #self.data['text'], self.data['word_count'] = list(zip(*self.data['Text'].map(process_rieger2021)))
        self.data['text'], self.data['word_count'] = list(zip(*[self.preprocess(text, nlp) for text, nlp in zipped]))
        self.data = self.data[self.data['text'] != '']
        self.data['label'] = self.data['white_supremacist'].astype(int)
        self.uniform_format()


class Rieger2021_white_supremacistDataset(Rieger2021Dataset):
    """ Annotated data from Rieger+ 2021 paper. This dataset just keeps items marked white supremacist. """

    def load(self):
        super().load()
        self.data = self.data.query('white_supremacist and Source==@self.source')

    def process(self):
        """ Remove label column """
        super().process()
        self.data.drop(columns='label', inplace=True)


class Hatecheck_identity_nonhateDataset(Dataset):
    """ Data selected from HateCheck to test lexical bias against marginalized identities """

    def load(self):
        self.data = pd.read_csv(self.load_paths[0], index_col=0)
        selected_identities = ['trans people', 'gay people', 'women',
                        'black people', 'Muslims', 'immigrants']
        selected_cols = ['ident_neutral_nh', 'ident_pos_nh', 
                        'slur_homonym_nh', 'slur_reclaimed_nh',
                        'negate_neg_nh',
                        'counter_quote_nh', 'counter_ref_nh',
                        ]
        self.data = self.data.query('target_ident==@selected_identities and functionality==@selected_cols')

    def process(self):
        self.data['text'], self.data['word_count'] = list(zip(*self.data.test_case.map(tokenize_lowercase)))
        self.data['label'] = 0
        self.uniform_format()


class Hatecheck_sampleDataset(Dataset):
    """ Data selected from HateCheck:
        * hate: that fits a white supremacist ideology
        * nonhate: to test lexical bias against marginalized identities 
    """

    def load(self):
        self.data = pd.read_csv(self.load_paths[0], index_col=0)
        selected_identities = ['trans people', 'gay people', 'women',
                        'black people', 'Muslims', 'immigrants']
        selected_cols = [
                        # Hate
                        'derog_neg_emote_h', 'derog_neg_attrib_h', 'derog_dehum_h',
                        'derog_impl_h', 'threat_dir_h', 'threat_norm_h', 'slur_h',
                        'refs_subs_clause_h', 'refs_subs_sent_h', 'negate_pos_h',
                        'phrase_question_h', 'phrase_opinion_h', 
                        'spell_char_swap_h', 'spell_char_del_h', 'spell_space_del_h', 'spell_space_add_h', 'spell_leet_h',

                        # Non-hate
                        'ident_neutral_nh', 'ident_pos_nh', 
                        'slur_homonym_nh', 'slur_reclaimed_nh',
                        'negate_neg_nh',
                        'counter_quote_nh', 'counter_ref_nh',
                        ]
        self.data = self.data.query('target_ident==@selected_identities and functionality==@selected_cols')

    def process(self):
        self.data['text'], self.data['word_count'] = list(zip(*self.data.test_case.map(tokenize_lowercase)))
        self.data['label'] = self.data.label_gold.map({'hateful': 1, 'non-hateful': 0})
        self.uniform_format()


class Medium_antiracistDataset(Dataset):
    """ Medium articles/blog posts scraped if they contained anti-racist tags """

    def load(self):
        dfs = []
        for fname in [f for f in os.listdir(self.load_paths[0]) if f.endswith('.jsonl')]:
            tag = re.split(r'_articles\d?\.', fname)[0]
            fpath = os.path.join(self.load_paths[0], fname)
            dfs.append(pd.read_json(fpath, orient='records', lines=True).assign(tag=tag))
        data = pd.concat(dfs).reset_index(drop=True)

        # Filter out empty articles
        self.data = data[data.title != data.text]

    def process(self):
        """ Sample data to match the number of long-form articles in the white supremacist corpus """
        ws_match = [key for key in self.lookup.keys() if 'white_supremacist' in key][0]
        self.data = self.data.sample(len(self.ref_corpora[ws_match]), random_state=9)

        # Process
        self.data['text'], self.data['word_count'] = list(zip(*tqdm(self.data['text'].map(tokenize_lowercase), ncols=60)))
        self.uniform_format(timestamp_col='date')


class Lda_annotationsDataset(Dataset):
    """ Posts from white supremacist corpus that I annotated for white supremacy (by LDA topic) """
    
    def process(self):
        """ Are already tokenized and preprocessed """
        self.data['word_count'] = self.data.text.str.split().str.len()
        self.data['label'] = self.data['ws']
        self.uniform_format() # Note that the source and domain will be wrong (but I don't think it matters)
