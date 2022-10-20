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

    # Settings
    subs = ['racism']

    api = PushshiftAPI()

    # PushShift logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger = logging.getLogger('psaw')
    logger.setLevel(logging.INFO)
    # To stop getting output, logger.setLevel(logging.ERROR)
    logger.addHandler(handler)

    # Scrape subreddit
    for subreddit in subs:
        outpath = f'../../data/antiracist/reddit_comments/{subreddit}.json'
        if not os.path.exists(os.path.dirname(outpath)):
            pdb.set_trace("No dir")
        print(subreddit)
        #dfs = []
        post_filter_list = ['id', 'selftext', 'title', 'author', 'created_utc', 'num_comments', 'score', 'brand_safe', 'over_18', 'domain', 'url', 'permalink']
        comment_filter_list = ['id', 'parent_id', 'body', 'author', 'created_utc', 'score', 'permalink']
        debug_limit = 100

        #posts = [post.d_ for post in api.search_submissions(
        #        subreddit=subreddit, after=row.begin, before=row.end, filter=post_filter_list, limit=int(row.post_count*.7))]
        #comments = [comment.d_ for comment in api.search_comments(
        #        subreddit=subreddit, after=row.begin, before=row.end, filter=comment_filter_list, limit=int(row.post_count*.7))]
        comments = [comment.d_ for comment in api.search_comments(subreddit=subreddit, filter=comment_filter_list)]
        # Remove deleted posts
        #if len(posts) == 0 and len(comments) == 0:
        if len(comments) == 0:
            tqdm.write('\tskip')
            continue
        #posts_df = pd.DataFrame([el for el in posts if el['selftext'] not in ['[deleted]', '[removed]']]).assign(post_type='submission')
        #comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']]).assign(post_type='comment')
        comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']])
        comments_df['created_utc'] = pd.to_datetime(comments_df.created_utc, unit='s')
        #dfs.append(comments_df)

        print(f'Finished, saving full dataset out to {outpath}')
        #data = pd.concat(dfs).reset_index(drop=True)
        data = comments_df
        print(f'{len(data)} comments scraped')
        data.to_json(outpath, orient='table', indent=4)


if __name__ == '__main__':
    main()
