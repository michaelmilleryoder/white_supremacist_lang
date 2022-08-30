#!/usr/bin/env python
# coding: utf-8
""" Get Reddit posts as examples of non-white supremacist forum talk, using Pushshift API """

from psaw import PushshiftAPI
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import pdb


def main():

    api = PushshiftAPI()

    # Load white supremacist dataset to count posts over time
    print('Loading white supremacist dataset to get post counts over time...')
    path = '../tmp/white_supremacist_corpus.pkl'
    ws_data = pd.read_pickle(path)

    # Select forum data, Group by year
    yearly = ws_data.query('domain=="forum"').groupby(by=ws_data.timestamp.dt.year)['text'].count()
    lookup = pd.DataFrame(yearly)
    lookup['begin'] = pd.to_datetime(yearly.index.astype(int).astype(str), format='%Y')
    lookup['end'] = lookup.begin.shift(-1, fill_value = datetime.datetime(2020,1,1))
    lookup.index.name = 'year'
    lookup.index = lookup.index.astype(int)
    lookup.rename(columns={'text': 'post_count'}, inplace=True)

    #start_year = 2013
    #lookup = lookup.loc[start_year:,]

    # Scrape subreddit
    subreddit = 'politics'
    dfs = []
    post_filter_list = ['id', 'selftext', 'title', 'author', 'created_utc', 'num_comments', 'score', 'brand_safe', 'over_18', 'domain', 'url', 'permalink']
    comment_filter_list = ['id', 'parent_id', 'body', 'author', 'created_utc', 'score', 'permalink']
    debug_limit = 100

    for index, row in tqdm(lookup.iterrows(), total=len(lookup)):
        tqdm.write(f'{index}, requesting {row.post_count} comments')
        #posts = [post.d_ for post in api.search_submissions(
        #        subreddit=subreddit, after=row.begin, before=row.end, filter=post_filter_list, limit=int(row.post_count*.7))]
        comments = [comment.d_ for comment in api.search_comments(
                subreddit=subreddit, after=row.begin, before=row.end, filter=comment_filter_list, limit=int(row.post_count*1.2))]
        # Remove deleted posts
        #if len(posts) == 0 and len(comments) == 0:
        if len(comments) == 0:
            tqdm.write('\tskip')
            continue
        #posts_df = pd.DataFrame([el for el in posts if el['selftext'] not in ['[deleted]', '[removed]']]).assign(post_type='submission')
        #comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']]).assign(post_type='comment')
        comments_df = pd.DataFrame([el for el in comments if el['body'] not in ['[deleted]', '[removed]']])
        #print(f'\tRemoved {len(posts)-len(posts_df)} deleted posts')
        #print(f'\tRemoved {len(comments)-len(comments_df)} deleted comments')
        #df = pd.concat([posts_df, comments_df]).sample(min(len(posts_df)+len(comments_df), row.post_count))
        df = comments_df.sample(min(len(comments_df), row.post_count))
        #selected = [x for x in ['title', 'selftext', 'body'] if x in df.columns]
        #selected = [x for x in ['title', 'body'] if x in df.columns]
        # df['text'] = sum(df[x].fillna('').astype(str) for x in selected) # doesn't work
        #df['text'] = ''
        #for colname in selected:
        #    df['text'] += df[colname].fillna('')
        #tqdm.write(f'\tSampled {len(df)} comments')
        dfs.append(df)

        # Save out for incremental results
        outpath = f'../tmp/{index}_{subreddit}_subreddit_comments.csv'
        df.to_csv(outpath)

    print('Finished, saving full dataset out')
    data = pd.concat(dfs).reset_index(drop=True)
    print(len(data))
    data.to_csv(f'../data/{subreddit}_subreddit_comments.csv')


if __name__ == '__main__':
    main()
