""" Storing and processing datasets, including processed versions 
    This module includes a superclass and subclasses for loading specific datasets
"""

import os
import json
import re
import pdb

import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

from utils import remove_mentions, remove_urls, tokenize_lowercase, process_4chan


class Dataset:
    """ Superclass for storing data and attributes of datasets """

    def __new__(cls, name, source, domain, load_paths):
        """ Choose the right subclass based on name """
        subclass_name = f"{name.capitalize()}Dataset"
        subclass_map = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
        subclass = subclass_map.get(subclass_name, cls)
        instance = super(Dataset, subclass).__new__(subclass)
        return instance

    def __init__(self, name, source, domain, load_paths):
        """ Args:
                name: dataset name
                source: source platform
                domain: discourse type of the source (tweet, chat, forum, long-form (article or book))
                load_paths: list of arguments for loading the datasets (like file paths)
        """
        self.name = name
        self.source = source
        self.domain = domain
        #self.loader = getattr(load_process_dataset, f'{self.name.capitalize()}Loader')
        self.load_paths = load_paths
        self.n_jobs = 20 # number of processes for multiprocessing of data
        self.data = pd.DataFrame()
        #cls.get_subclass()

    def uniform_format(self, timestamp_col=None, unit=None, errors='raise'):
        """ Format the dataframe for combining with other datasets
            Set a column to a timestamp data type if there is one
            Create an index for the dataset with the dataset name.
            Add a column of the dataset name
            Filter self.data to just the columns needed from all datasets:
            Args:
                timestamp_col: the name of a column to convert to a datetime data type. If None, no timestamp
                unit: Additional parameter to pass to pd.to_datetime
        """

        if timestamp_col is not None:
            self.data['timestamp'] = pd.to_datetime(self.data[timestamp_col], utc=True, unit=unit, errors=errors)
        self.data['id'] = self.name + '_' + self.data.index.astype(str)
        self.data.set_index('id', inplace=True)
        self.data['dataset'] = self.name
        self.data['source'] = self.source
        self.data['domain'] = self.domain
        if 'timestamp' in self.data.columns:
            self.data = self.data[['text', 'timestamp', 'dataset', 'source']]
        else:
            self.data = self.data[['text', 'dataset', 'source']]

    def load(self):
        """ Needs to be overridden in a subclass """
        self.data = pd.read_csv(self.load_fpaths[0])

    def process(self):
        """ Needs to be overridden in a subclass """
        pass


class Qian2018Dataset(Dataset):

    @staticmethod
    def process_text(text, user_mentions, urls, tokenizer):
        new_text = text
        if isinstance(user_mentions, list):
            new_text = remove_mentions(new_text, user_mentions)
        if isinstance(urls, list):
            new_text = remove_urls(new_text, urls)
        new_text = ' '.join(tokenizer.tokenize(new_text))
        return new_text.lower()

    def load(self):
        """ Load tweets """
        with open(self.load_paths[0], 'r') as f:
            self.data = pd.json_normalize([json.loads(tweet) for tweet in f.read().splitlines()])

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Tokenize, anonymize texts
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['processed_text'] = [self.process_text(
                text, user_mentions, urls, tokenizer) for text, user_mentions, urls in tqdm(zip(
                self.data['text'], self.data['entities.mentions'], self.data['entities.urls']), total=len(self.data), ncols=80)]
        self.data.drop(columns='text', inplace=True)
        self.data.rename(columns={'id': 'tweet_id', 'processed_text': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='created_at')


class Elsherief2021Dataset(Dataset):

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
        # May have duplicates (and could remove them if I match them with tweet ids)
        self.data = pd.concat([white_grievance, user_matches]).drop_duplicates(subset='id').reset_index(drop=True)
        self.data.rename(columns={'id': 'tweet_id'}, inplace=True)
        
    def process(self):
        """ Process data into a format to combine with other datasets """
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['processed_text'] = [Qian2018Dataset.process_text(
                text, user_mentions, urls, tokenizer) for text, user_mentions, urls in tqdm(zip(
                self.data['text'], self.data['entities.mentions'], self.data['entities.urls']), total=len(self.data), ncols=80)]
        self.data.drop(columns='text', inplace=True)
        self.data.rename(columns={'processed_text': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='created_at')


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
        self.data = pd.concat(message_dfs)

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Remove any common first names
        names = pd.read_csv(self.load_paths[-1], skiprows=[0])
        names = names.query('Rank <= 300')
        common_names = set(names['Name'].str.lower()).union(names['Name.1'].str.lower())
        # Remove spencer, guy (for Richard Spencer)
        common_names -= {'spencer', 'guy'}
        self.data['processed'] = self.data['message'].map(lambda x: ' '.join([wd for wd in nltk.word_tokenize(str(x)) if wd not in common_names]).lower())
        self.data.rename(columns={'processed': 'text', inplace=True)
        self.uniform_format(timestamp_col='timestamp')


class IronmarchDataset(Dataset):

    def process():
        """ Process data into a format to combine with other datasets """
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(tokenize_lowercase, self.data['index_content']), total=len(self.data)))
        self.data.reset_index(drop=True, inplace=True)
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='index_date_created', unit='s')


class StormfrontDataset(Dataset):

    @staticmethod
    def preprocess(inp):
        text = re.sub(r'Quote:\n\n\n\n\nOriginally Posted by .*\n\n\n', '', inp) # Remove quote tag
        text = re.sub(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', '', text) # Remove URLs
        text = ' '.join(nltk.word_tokenize(str(text))).lower()
        return text

    def load():
        """ Load data dump """
        dfs = []
        for fname in tqdm(os.listdir(self.load_paths[0]), ncols=80):
            fpath = os.path.join(dirpath, fname)
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

    def process():
        """ Process data into a format to combine with other datasets """
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(preprocess, self.data['text']), total=len(self.data)))
        self.data.drop(columns='text', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data.rename(columns={'processed': 'text'})
        self.uniform_format(timestamp_col='timestamp', errors='coerce')


class Jokubausaite2020Dataset(Dataset):

    def load():
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
            dfs.append(pd.read_csv(fpath))

        self.data = pd.concat(dfs).reset_index(drop=True)

    def process():
        """ Process data into a format to combine with other datasets """
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(process_4chan, self.data.body), total=len(self.data), ncols=80))
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='timestamp')
