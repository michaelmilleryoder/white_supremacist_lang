#!/usr/bin/env python
# coding: utf-8
""" Get Reddit posts as examples of non-white supremacist forum talk, using Pushshift API """

import os
from psaw import PushshiftAPI
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import pdb
import logging


def main():

    api = PushshiftAPI()

    # PushShift logging
    #handler = logging.StreamHandler()
    #handler.setLevel(logging.INFO)
    #logger = logging.getLogger('psaw')
    #logger.setLevel(logging.INFO)
    ## To stop getting output, logger.setLevel(logging.ERROR)
    #logger.addHandler(handler)

    # Settings
    subs = ['BlackLivesMatter', 'racism', 'StopAntiAsianRacism']
    out_dirpath = '../../data/antiracist/reddit_comments/'

    # Load white supremacist dataset to count posts over time
    print('Loading (training) white supremacist dataset to get post counts over time...')
    path = '../../data/corpora/white_supremacist_train_corpus.json'
    ws_data = pd.read_json(path, orient='table')

    # Select forum data, Group by year
    yearly = ws_data.query('domain=="forum"').groupby(by=ws_data.timestamp.dt.year)['text'].count()
    lookup = pd.DataFrame(yearly)
    lookup['begin'] = pd.to_datetime(yearly.index.astype(int).astype(str), format='%Y')
    lookup['end'] = [x.replace(year=x.year + 1) for x in lookup['begin']]
    lookup.index.name = 'year'
    lookup.index = lookup.index.astype(int)
    lookup.rename(columns={'text': 'post_count'}, inplace=True)

    #start_year = 2013
    #lookup = lookup.loc[start_year:,]

    #subreddit_limits = {
    #    'politics': 1.0
    #    'Europe': 0.5,
    #    'USA': 0.7,
    #    'AskAnAmerican': 0.5
    #}

    # Scrape subreddit
    #for subreddit in ['politics', 'Europe', 'USA', 'AskAnAmerican']:
    for subreddit in subs:
        outpath = os.path.join(out_dirpath, f'{subreddit}.json')
        if not os.path.exists(os.path.dirname(outpath)):
            tqdm.write("No dir")
        print(subreddit)
        dfs = []
        post_filter_list = ['id', 'selftext', 'title', 'author', 'created_utc', 'num_comments', 'score', 'brand_safe', 'over_18', 'domain', 'url', 'permalink']
        comment_filter_list = ['id', 'parent_id', 'body', 'author', 'created_utc', 'score', 'permalink']
        debug_limit = 100

        for index, row in tqdm(lookup.iterrows(), total=len(lookup)):
            tqdm.write(f'{index}, requesting {row.post_count} comments')
            #posts = [post.d_ for post in api.search_submissions(
            #        subreddit=subreddit, after=row.begin, before=row.end, filter=post_filter_list, limit=int(row.post_count*.7))]
            comments = [comment.d_ for comment in api.search_comments(
                    subreddit=subreddit, after=row.begin, before=row.end, filter=comment_filter_list, limit=int(row.post_count*.7))]
            # Remove deleted posts
            #if len(posts) == 0 and len(comments) == 0:
            if len(comments) == 0:
                tqdm.write('\tskip')
                continue
            #posts_df = pd.DataFrame([el for el in posts if el['selftext'] not in ['[deleted]', '[removed]']]).assign(post_type='submission')
            #comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']]).assign(post_type='comment')
            comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']])
            comments_df['created_utc'] = pd.to_datetime(comments_df.created_utc, unit='s')
            #print(f'\tRemoved {len(posts)-len(posts_df)} deleted posts')
            #print(f'\tRemoved {len(comments)-len(comments_df)} deleted comments')
            #df = pd.concat([posts_df, comments_df]).sample(min(len(posts_df)+len(comments_df), row.post_count))
            #df = comments_df.sample(min(len(comments_df), row.post_count))
            #selected = [x for x in ['title', 'selftext', 'body'] if x in df.columns]
            #selected = [x for x in ['title', 'body'] if x in df.columns]
            # df['text'] = sum(df[x].fillna('').astype(str) for x in selected) # doesn't work
            #df['text'] = ''
            #for colname in selected:
            #    df['text'] += df[colname].fillna('')
            #tqdm.write(f'\tSampled {len(df)} comments')
            dfs.append(comments_df)

            # Save out for incremental results
            #outpath = f'../tmp/{index}_{subreddit}_subreddit_comments.pkl'
            #df.to_pickle(outpath)

        print('Finished, saving full dataset out')
        data = pd.concat(dfs).reset_index(drop=True)
        print(len(data))
        data.to_json(outpath, orient='table', indent=4)


if __name__ == '__main__':
    main()
