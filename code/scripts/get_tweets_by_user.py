#!/usr/bin/env python
# coding: utf-8

""" Get tweets from antiracist user lists """

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

    # Load username list
    #print("Getting query terms from white supremacist dataset...")
    list_name = 'unc'
    path = '../../resources/unc_antiracist_twitter.txt'
    with open(path) as f:
        usernames = f.read().splitlines()
    start_name = 'ShowUp4RJ' # if restarting somewhere
    if start_name is not None:
        usernames = usernames[usernames.index(start_name):]

    print("Getting tweets from usernames...")
    # Load authentication
    keys = pd.read_csv('../../../tweepy_oauth_academic.csv', index_col='name').to_dict()['key']
    client = tweepy.Client(keys['bearer_token'], wait_on_rate_limit=True)

    tweet_fields = [
        'id', 'created_at', 'text', 'author_id', 'conversation_id', 'entities', 'public_metrics', 
        'geo', 'lang', 'referenced_tweets'
    ]
    user_fields = [
        'id', 'name', 'username', 'description'
    ]

    out_dirpath = '../../data/antiracist/twitter'
    if not os.path.exists(out_dirpath):
        raise ValueError("Output directory doesn't exist")
    debug_max_results = 10
    for username in tqdm(usernames, ncols=80):
        fetched = []
        next_token = None
        while True:
            try:
                response = client.search_all_tweets(
                        f'from:{username}', 
                        tweet_fields=tweet_fields, 
                        user_fields=user_fields,
                        max_results=500,
                        #max_results=debug_max_results,
                        start_time=datetime(2010,1,1),
                        next_token=next_token,
                        )
                if response.data is not None:
                    fetched.append(response)
            except tweepy.BadRequest as e:
                tqdm.write(str(e))
                tqdm.write(f'Bad request: {username}')
            except tweepy.errors.TwitterServerError as e:
                tqdm.write(str(e))
                tqdm.write("Waiting 10 minutes and trying again...")
                time.sleep(600) # Wait 10 minutes and try again
            if 'next_token' in response.meta:
                next_token = response.meta['next_token']
            else:
                break
            time.sleep(1)
            tqdm.write(f"Received {len(fetched)} responses so far (scraped username {username})")
        try:
            tweets = [tweet.data for response in fetched for tweet in response.data if response.data is not None]
        except Exception as e:
            pdb.set_trace()

        # Save out tweet data for username
        if len(tweets) > 0:
            outpath = os.path.join(out_dirpath, f'{username}_tweets.jsonl')
            with open(outpath, 'w') as f:
                f.write('\n'.join([json.dumps(tweet) for tweet in tweets]) + '\n')
            tqdm.write(f"Saved out data to {outpath}")


if __name__ == '__main__':
    main()
