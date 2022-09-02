#!/usr/bin/env python
# coding: utf-8

# # Get tweets using the API from tweet IDs

from tqdm import tqdm
import json
import os
import tweepy
import pandas as pd

# Settings
#dataset = 'qian2018'
dataset = 'elsherief2021'

# Load authentication
keys = pd.read_csv('/storage2/mamille3/tweepy_oauth_academic.csv', index_col='name').to_dict()['key']
client = tweepy.Client(keys['bearer_token'], wait_on_rate_limit=True)

# Load Qian+2018 tweet IDs
#fpath = '/storage2/mamille3/data/hate_speech/qian2018/white_supremacist_tweets.csv'
#tweet_ids = pd.read_csv(fpath)['tweet id']

# Load ElSherief+2021 tweet IDs
stg1_meta = pd.read_csv('/storage2/mamille3/data/hate_speech/elsherief2021/implicit_hate_v1_stg1.tsv', sep='\t')
stg1_meta_tweets = stg1_meta[~stg1_meta['ID'].str.contains('_')].copy()
tweet_ids = stg1_meta_tweets['ID']

# Break up into lists of 100
chunks = [tweet_ids[x:min(x+100, len(tweet_ids))].values.tolist() for x in range(0, len(tweet_ids), 100)]

# Do query and save results out
tweet_fields = [
    'id', 'created_at', 'text', 'author_id', 'conversation_id', 'entities', 'public_metrics', 'geo', 'lang', 'referenced_tweets'
]
user_fields = [
    'id', 'name', 'username', 'description'
]

out_dirpath = os.path.join('/storage2/mamille3/white_supremacist_lang/data', dataset)
if not os.path.exists(out_dirpath):
    os.mkdir(out_dirpath)

for chunk in tqdm(chunks, ncols=80):
    response = client.get_tweets(chunk, expansions=['author_id', 'entities.mentions.username'], tweet_fields=tweet_fields, user_fields=user_fields)
    
    # Save out tweet data
    outpath = os.path.join(out_dirpath, 'data.jsonl')
    tweets = [tweet.data for tweet in response.data]
    with open(outpath, 'a') as f:
        f.write('\n'.join([json.dumps(tweet) for tweet in tweets]) + '\n')

    # Save out users
    outpath = os.path.join(out_dirpath, 'users.jsonl')
    users = [user.data for user in response.includes['users']]
    with open(outpath, 'a') as f:
        f.write('\n'.join([json.dumps(user) for user in users]) + '\n')

    # Save out errors
    outpath = os.path.join(out_dirpath, 'errors.jsonl')
    with open(outpath, 'a') as f:
        f.write('\n'.join([json.dumps(error) for error in response.errors]) + '\n')
