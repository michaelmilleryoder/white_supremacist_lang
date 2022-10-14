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

from utils import (remove_mentions, remove_urls, tokenize_lowercase, process_4chan,
        process_tweet, process_tweet_text, process_article, process_reddit, process_chat, 
        load_now, process_now, process_rieger2021)


def ref_corpus_year_count(ref_corpus):
    """ Returns a table of post and word counts per year in a reference corpus (for matching samples) 
        ref_corpus: pandas DataFrame of data used as a reference corpus
    """
    yearly = ref_corpus.groupby(by=ref_corpus.timestamp.dt.year).agg({
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

    def __new__(cls, name, source, domain, load_paths, ref_corpus=None):
        """ Choose the right subclass based on dataset name """
        subclass_name = f"{name.capitalize()}Dataset"
        #subclass_map = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
        subclass_map = {subclass.__name__: subclass for subclass in cls.all_subclasses()}
        #subclass = subclass_map.get(subclass_name.split('_')[0], cls)
        subclass = subclass_map.get(subclass_name, cls)
        instance = super(Dataset, subclass).__new__(subclass)
        return instance

    def __init__(self, name, source, domain, load_paths, ref_corpus=None):
        """ Args:
                name: dataset name
                source: source platform
                domain: discourse type of the source (tweet, chat, forum, long-form (article or book))
                load_paths: list of arguments for loading the datasets (like file paths)
                ref_corpus: a dataframe of another corpus to use as a reference (for sampling a similar size, e.g.)
        """
        self.name = name
        self.source = source
        self.domain = domain
        #self.loader = getattr(load_process_dataset, f'{self.name.capitalize()}Loader')
        self.load_paths = load_paths
        self.ref_corpus = ref_corpus
        if self.ref_corpus is not None:
            self.ref_corpus = ref_corpus.query(f'domain=="{domain}"').copy()
            self.ref_corpus['word_count'] = self.ref_corpus.text.str.split().str.len()
            self.lookup = ref_corpus_year_count(self.ref_corpus)
        self.n_jobs = 20 # number of processes for multiprocessing of data
        self.data = pd.DataFrame()
        #cls.get_subclass()

    def uniform_format(self, timestamp_col=None, unit=None, errors='raise', format=None):
        """ Format the dataframe for combining with other datasets
            Set a column to a timestamp data type if there is one
            Create an index for the dataset with the dataset name.
            Add a column of the dataset name, source and domain.
            Filter self.data to just the columns needed from all datasets:
                text, timestamp (if present), dataset, source, domain
            Args:
                timestamp_col: the name of a column to convert to a datetime data type. If None, no timestamp
                unit: Additional parameter to pass to pd.to_datetime
                format: Additional parameter to pass to pd.to_datetime
        """

        if timestamp_col is not None:
            self.data['timestamp'] = pd.to_datetime(self.data[timestamp_col], format=format, utc=True, unit=unit, errors=errors)
        self.data['id'] = self.name + '_' + self.data.index.astype(str)
        self.data.set_index('id', inplace=True)
        self.data['dataset'] = self.name
        self.data['source'] = self.source
        self.data['domain'] = self.domain
        selected_cols = ['text', 'dataset', 'source', 'domain'] + [col for col in ['timestamp', 'label'] if col in self.data.columns]
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

        # Comparison to matching white supremacist data
        self.data['word_count'] = self.data.text.str.split().str.len()
        print(f'\t\tMatching white supremacist corpus word count mean: {self.ref_corpus.word_count.mean()}')
        print(f'\t\tNeutral word count mean: {self.data.word_count.mean()}')
        print(f'\t\tMatching white supremacist corpus word count sum: {self.ref_corpus.word_count.sum()}')
        print(f'\t\tNeutral word count sum: {self.data.word_count.sum()}')


class Qian2018Dataset(Dataset):

    def load(self):
        """ Load tweets """
        with open(self.load_paths[0], 'r') as f:
            self.data = pd.json_normalize([json.loads(tweet) for tweet in f.read().splitlines()])

    def process(self):
        """ Process data into a format to combine with other datasets """
        # Tokenize, anonymize texts
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['processed_text'] = [process_tweet(
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
        self.data['processed_text'] = [process_tweet(
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
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='timestamp')


class IronmarchDataset(Dataset):

    def process(self):
        """ Process data into a format to combine with other datasets """
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(
                    tokenize_lowercase, self.data['index_content']), total=len(self.data), ncols=80))
        self.data.reset_index(drop=True, inplace=True)
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='index_date_created', unit='s')


class StormfrontDataset(Dataset):

    @classmethod
    def preprocess(cls, inp):
        text = re.sub(r'Quote:\n\n\n\n\nOriginally Posted by .*\n\n\n', '', inp) # Remove quote tag
        text = re.sub(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', '', text) # Remove URLs
        text = ' '.join(nltk.word_tokenize(str(text))).lower()
        return text

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
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(self.preprocess, self.data['text']), total=len(self.data), ncols=80))
        self.data.drop(columns='text', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='timestamp', errors='coerce')


class Jokubausaite2020Dataset(Dataset):

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
        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(process_4chan, self.data.body), total=len(self.data), ncols=80))
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='timestamp')


class Papasavva2020Dataset(Dataset):

    def load(self):
        """ Load dataset """
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

        with Pool(self.n_jobs) as p:
            self.data['processed'] = list(tqdm(p.imap(process_4chan, self.data.body), total=len(self.data), ncols=80))
        self.data.rename(columns={'processed': 'text'}, inplace=True)

        self.uniform_format(timestamp_col='time', unit='s')


class Calderon2021Dataset(Dataset):

    def load(self):
        """ Load data dump """
        with open(self.load_paths[0]) as f:
            self.data = pd.json_normalize(json.load(f))

    def process(self):
        """ Process data into a format to combine with other datasets """
        with Pool(self.n_jobs) as p:
            self.data['text'] = list(tqdm(p.imap(
                    process_article, self.data['title'] + ' ' + self.data['author_wording']), total=len(self.data), ncols=80))
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

    def process(self): 
        """ Process data into a format to combine with other datasets """
        self.data['processed'] = list(map(tokenize_lowercase, tqdm(self.data['orig_text'], total=len(self.data), ncols=80)))
        self.data.rename(columns={'processed': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='year', format='%Y')


class Reddit_matchDataset(Dataset):
    """ Neutral (non-white supremacist) Reddit data that matches forum data in white supremacy dataset """

    def load(self):
        """ Load prescraped Reddit data (get_reddit.py) """
        fpaths = sorted([fname for fname in os.listdir(self.load_paths[0]) if fname.endswith('.json')])
        dfs = []
        for fname in tqdm(fpaths, ncols=80):
            # print(fname)
            fpath = os.path.join(self.load_paths[0], fname)
            sub = pd.read_json(fpath, orient='table')
            subreddit = fname.split('_')[0]
            dfs.append(sub.assign(subreddit=subreddit.lower()))
        self.data = pd.concat(dfs)
        #self.data['year'] = self.data.created_utc.dt.year

    def process(self):
        """ Sample data to match forum data in white supremacist data.
            Process data for combining with other datasets in neutral corpus.
        """
        self.data = self.data.groupby(self.data.created_utc.dt.year).apply(
                lambda group: group.sample(self.lookup[('post_count', 'count')][group.name])).reset_index(drop=True)
        with Pool(self.n_jobs) as p:
            self.data['text'] = list(tqdm(p.imap(
                    process_reddit, self.data['body']), total=len(self.data), ncols=80))
        self.uniform_format(timestamp_col='created_utc')

    #def print_stats(self):
    #    """ Print statistics on number of posts and word count per year compared to the reference corpus
    #        TODO: need to get subreddit proportions before uniform_format
    #    """

    #    # Subreddit proportions
    #    sub_distro = pd.concat([self.data.subreddit.value_counts(), self.data.subreddit.value_counts(normalize=True)], axis=1)
    #    sub_distro.columns = ['post_count', 'proportion']
    #    print(sub_distro.to_string())

    #    # Comparison to matching white supremacist data
    #    post_counts = pd.concat([self.lookup['post_count'], 
    #        self.data.groupby(self.data.created_utc.dt.year).text.count()], axis=1)
    #    print(post_counts.to_string())

    #    super().print_stats()


class Discord_matchDataset(Dataset):
    """ Neutral (non-white supremacist) Discord data to match chat data in white supremacy dataset """
    
    def load(self):
        """ Load Discord random dataset """
        fpaths = [os.path.join(self.load_paths[0], fname) for fname in os.listdir(self.load_paths[0]) if fname.endswith('.txt')]
        dfs = []
        for fpath in tqdm(fpaths, ncols=80):
            with open(fpath) as f:
                dfs.append(pd.DataFrame({'message': [message for line in f.read().splitlines() for message in line.split('\t')]}))
        self.data = pd.concat(dfs)

    def process(self):
        """ Sample data to match chat data in white supremacist training dataset.
            Process data for combining with other datasets in neutral corpus.
        """

        # Sample to match the word count of white supremacist matching chat data
        self.data = self.data.sample(int(self.ref_corpus.word_count.sum()/4.2)) 
        # 4.2 is the average words/posts in the Discord data (see neutral_data.ipynb)

        # Process data
        tokenizer = TweetTokenizer(strip_handles=True)
        zipped = zip(self.data['message'], itertools.repeat(tokenizer))
        with Pool(self.n_jobs) as p:
            #self.data['text'] = list(tqdm(p.imap(process_chat, *zipped), total=len(self.data), ncols=80))
            self.data['text'] = p.starmap(process_chat, tqdm(zipped, total=len(self.data), ncols=80))
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

        with Pool(20) as p:
            dfs = list(tqdm(p.imap(load_now, fpaths), total=len(fpaths), ncols=80))
        self.data = pd.concat(dfs)

    def process(self):
        """ Sample data to match long-form data in white supremacist training dataset.
            Process data for combining with other datasets in neutral corpus.
        """
        # Sample specific number of articles by year
        self.data = self.data[self.data['year'].isin(self.lookup.index)].groupby('year').apply(
                lambda group: group.sample(int(self.lookup.loc[group.name, ('post_count', 'count')]/3.5))).reset_index(
                drop = True)
        # 3.5 is approximately the number of articles to match the word count in white supremacist data

        # Process data
        with Pool(self.n_jobs) as p:
            self.data['text'] = list(tqdm(p.imap(
                    process_now, self.data['article']), total=len(self.data), ncols=80))
        self.uniform_format(timestamp_col='year', format='%Y')


class Twitter_matchDataset(Dataset):
    """ Neutral (non-white supremacist) Twitter data to match Twitter data in white supremacy dataset """

    def load(self):
        """ Load tweets collected through keywords at get_tweets_by_query.py (I think) and get_tweets_by_query.ipynb """
        dfs = []
        for fname in tqdm(sorted(os.listdir(self.load_paths[0]))):
            with open(os.path.join(self.load_paths[0], fname)) as f:
                dfs.append(pd.json_normalize([json.loads(line) for line in f.read().splitlines()]))
        self.data = pd.concat(dfs).reset_index(drop=True)

    def process(self):
        """ Process data for combining with other datasets in neutral corpus.
        """
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['processed_text'] = [process_tweet(
                text, user_mentions, urls, tokenizer) for text, user_mentions, urls in tqdm(zip(
                self.data['text'], self.data['entities.mentions'], self.data['entities.urls']), total=len(self.data), ncols=80)]
        self.data.drop(columns='text', inplace=True)
        self.data.rename(columns={'id': 'tweet_id', 'processed_text': 'text'}, inplace=True)
        self.uniform_format(timestamp_col='created_at')


class Alatawi2021Dataset(Dataset):
    """ Tweets annotated for white supremacy from Alatawi+ 2021 paper """
    
    def process(self):
        """ Process data for evaluating classifiers based on other datasets. """
        self.data['text'] = self.data['input.text'].map(tokenize_lowercase)
        self.data['label'] = self.data['Voting and Final Labels']
        self.uniform_format()


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
            hs_nodups.query('hatespeech == "no"').rename(columns={'hatespeech': 'white_nationalism'}).sample(round(n_wn*7/3)),
        ]).sample(frac=1)

    def process(self):
        """ Process data for evaluating classifiers based on other datasets. """
        tokenizer = TweetTokenizer(strip_handles=True)
        self.data['text'] = [process_tweet_text(text, tokenizer) for text in self.data['text']]
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


class Adl_heatmapDataset(Dataset):
    """ Offline propaganda from ADL HEATMap dataset """

    def load(self):
        # Load quotes (already extracted from event descriptions)
        quotes = pd.read_csv(self.load_paths[0])
        quotes['timestamp'] = pd.to_datetime(quotes.date, format='%m/%d/%y', errors='coerce', utc=True).fillna(
                pd.to_datetime(quotes.date, format='%y-%b', errors='coerce', utc=True))

        # Filter to quotes from groups with white supremacist ideologies
        incidents = pd.read_csv(self.load_paths[1])
        incidents['timestamp'] = pd.to_datetime(incidents.date, utc=True)

        merged = pd.merge(quotes, incidents, how='left', on=['description', 'city', 'timestamp'])
        self.data = merged[merged['ideology']== 'Right Wing (White Supremacist)']
        
        # Remove duplicates
        self.data = self.data.drop_duplicates(subset='quote').reset_index(drop=True)

    def process(self):
        self.data['text'] = self.data['quote'].map(tokenize_lowercase)
        self.data['label'] = 1 # all labeled as white supremacist
        self.uniform_format()


class Rieger2021Dataset(Dataset):
    """ Annotated data from Rieger+ 2021 paper """

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
            1: 'td',
            2: '4chan_pol',
            3: '8chan_pol',
        })

        data['white_supremacist'] = data['inhuman_ideology'].isin(['white supremacy/white ethnostate', 'National Socialist'])

        # Select positive and negative examples
        self.data = data.query('white_supremacist or (gen_ins==0 and viol==0 and pers_ins==0 and not white_supremacist)')

    def process(self):
        # Remove NaNs of Text
        self.data = self.data.dropna(subset='Text')
        self.data['text'] = self.data['Text'].map(process_rieger2021)
        self.data = self.data[self.data['text'] != '']
        self.data['label'] = self.data['white_supremacist'].astype(int)
        self.uniform_format()


class Hatecheck_identity_nonhateDataset(Dataset):
    """ Data selected from HateCheck to test lexical bias against marginalized identities """

    def load(self):
        self.data = pd.read_csv(self.load_paths[0], index_col=0)
        selected_cols = ['ident_neutral_nh', 'ident_pos_nh']
        self.data = self.data.query('functionality==@selected_cols')

    def process(self):
        self.data['text'] = self.data.test_case.map(tokenize_lowercase)
        self.data['label'] = 0
        self.uniform_format()
