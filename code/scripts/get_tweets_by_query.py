#!/usr/bin/env python
# coding: utf-8

""" Get tweets for neutral (non-white supremacist) data """

import os
import json
import re
import tweepy
import pandas as pd
import datetime
import nltk
import string
from collections import Counter
import random
from datetime import datetime
from tqdm import tqdm
import pdb
import time


def main():

    print("Getting query terms from white supremacist dataset...")
    # Load white supremacist dataset to count tweets over time
    path = '../tmp/white_supremacist_corpus.pkl'
    ws_data = pd.read_pickle(path)

    # Select tweet data, Group by year
    yearly = ws_data.query('domain=="tweet/short propaganda"').groupby(by=ws_data.timestamp.dt.year)['text'].count()
    lookup = pd.DataFrame(yearly)
    lookup['begin'] = pd.to_datetime(yearly.index.astype(int).astype(str), format='%Y')
    lookup['end'] = [min(x.replace(year=x.year + 1), datetime.now()) for x in lookup['begin']]
    lookup.index.name = 'year'
    lookup.index = lookup.index.astype(int)
    lookup.rename(columns={'text': 'post_count'}, inplace=True)

    # ## Sample query words from white supremacist tweet data 
    # To get tweets that share terms but aren't likely white supremacist
    random.seed(9)
    stopwords = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['...', '…', '’', ';-)', 'rt', '<url>', '“', '”', "n't", 'new', 'report', 'border', 'via', '—', '):', '°', '‘', "'re"] 
    slurs_fpath = '/storage2/mamille3/data/hate_speech/hatebase_slurs.txt'
    with open(slurs_fpath) as f:
        slurs = f.read().splitlines()
        slurs += [term + 's' for term in slurs]
        slurs +=  ['mudshark', 'illegals', 'anti-white']
    def check_word(word):
        """ See if word is ok to be a query """
        return re.search('[a-zA-Z]', word) and not (word in stopwords or word in slurs or word.startswith('#') or word.startswith('http') or word.startswith('http') or '.' in word)

    words_by_year = ws_data.query('domain=="tweet/short propaganda"').groupby(by=ws_data.timestamp.dt.year).agg(
        {'text': lambda x: {w: count for w, count in Counter([w for w in ' '.join(x).split() if check_word(w)]).items() if count > 1}})
    words_by_year['text'] = words_by_year

    lookup['total_words'] = words_by_year
    lookup['sampled_words'] = [Counter(random.choices(list(ctr.keys()), weights=list(ctr.values()), k=int(n/10))).most_common() for ctr, n in zip(lookup.total_words, lookup.post_count)]
    lookup.drop(columns='total_words', inplace=True)

    print("Getting tweets from queries...")
    # ## Get tweets from sampled words
    # Load authentication
    keys = pd.read_csv('/storage2/mamille3/tweepy_oauth_academic.csv', index_col='name').to_dict()['key']
    client = tweepy.Client(keys['bearer_token'], wait_on_rate_limit=True)

    tweet_fields = [
        'id', 'created_at', 'text', 'author_id', 'conversation_id', 'entities', 'public_metrics', 'geo', 'lang', 'referenced_tweets'
    ]
    #place_fields = [
    #    'full_name', 'id', 'contained_within'
    #]

    start_year = 2016
    lookup = lookup.loc[start_year:]

    for i, row in lookup.iterrows():
        tqdm.write(str(i))
        fetched = []
        n_queries = len(row.sampled_words)
        #for j, (word, count) in enumerate(tqdm(row.sampled_words, total=n_queries, ncols=80):
        for j, (word, count) in enumerate(tqdm(row.sampled_words, ncols=80)):
            try:
                response = client.search_all_tweets(
                        word + ' lang:en', 
                        tweet_fields=tweet_fields, 
                        start_time=row.begin, 
                        end_time=row.end, 
                        max_results=count*10)
            except tweepy.BadRequest as e:
                tqdm.write(str(e))
                tqdm.write(f'Bad request: {word}')
            time.sleep(1)
            #tqdm.write(f'{j} requests done')
        tweets = [tweet.data for response in fetched for tweet in response.data if response.data is not None]

        # Save out tweet data
        out_dirpath = '../data/neutral/twitter'
        outpath = os.path.join(out_dirpath, f'{row.begin.year}_data.jsonl')
        with open(outpath, 'w') as f:
            f.write('\n'.join([json.dumps(tweet) for tweet in tweets]) + '\n')
        tqdm.write(f"Saved out data to {outpath}")


if __name__ == '__main__':
    main()
