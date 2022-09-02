""" Script to process web archive of Stormfront into pandas dataframes of posts """

import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import pdb
#import dateutil.parser as dparser
import numpy as np


dfs = []
base_dirpath = '/storage2/mamille3/white_supremacist_lang/data/stormfront_archive/stormfront.org_201708/'


def extract_dir(dirpath):
    """ Extract posts from a directory of an archive """
    dfs = []
    forum_dirpath = os.path.join(base_dirpath, dirpath, 'www.stormfront.org', 'forum')
    if not os.path.exists(forum_dirpath):
        return
    outpath = f'/storage2/mamille3/white_supremacist_lang/data/stormfront_archive/processed/{dirpath}.csv'
    if os.path.exists(outpath):
        tqdm.write("exists: skipping")
        return
    tqdm.write(dirpath)
    dirs = [fname for fname in sorted(os.listdir(forum_dirpath)) if re.match(r't\d', fname)]
    for dirname in tqdm(dirs):
    #for dirname in dirs:
        for fname in os.listdir(os.path.join(forum_dirpath, dirname)):
            fpath = os.path.join(forum_dirpath, dirname, fname)
            # print(fpath)
            with open(fpath, encoding='latin-1') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                dfs.append(get_posts(soup))
    messages = pd.concat(dfs)

    # Remove unicode surrogates
    messages['text'] = messages['text'].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))

    messages.to_csv(outpath, index=False)
    tqdm.write("csv written")
    return


def get_posts(soup):
    """ Returns a df of posts from a forum page (any timestamp?) """
    # Get forum title (breadcrumbs)
    breadcrumb = ' '.join([el.get_text() for el in soup.find_all('span', class_='navbar')])
    posts = soup.find(id='posts')
    if posts is None:
        return
    messages = posts.find_all(id = re.compile(r'post_message_\d+'))
    post_ids = [message['id'].split('_')[-1] for message in messages]
    post_texts = [message.get_text().strip() for message in messages]
    # timestamps = [dparser.parse(soup.find('a', {'name': f'post{post_id}'}).next_sibling.strip()) for post_id in post_ids]
    timestamps = [soup.find('a', {'name': f'post{post_id}'}).next_sibling.strip() for post_id in post_ids]

    # Assemble into a df
    df = pd.DataFrame({'post_id': post_ids, 'thread_breadcrumb': breadcrumb, 'timestamp': timestamps, 'text': post_texts})
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='ignore')
    return df


def main():
    dirpaths = sorted([dirpath for dirpath in os.listdir(base_dirpath) if os.path.isdir(os.path.join(base_dirpath, dirpath)) and dirpath.startswith('stormfront')])
    with Pool(20) as p:
        list(tqdm(p.imap(extract_dir, dirpaths), total=len(dirpaths), ncols=80))
        # tqdm progress bar doesn't work for some reason
    # Debugging
    # for dirpath in dirpaths[:1]:
    #     messages = extract_dir(dirpath)


if __name__ == '__main__':
    main()
