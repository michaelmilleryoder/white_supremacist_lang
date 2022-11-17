""" Utility functions for processing data, etc 

    @author Michael Miller Yoder
    @year 2022
"""

import re
import os
import html
from html.parser import HTMLParser
from datetime import datetime

import pandas as pd
import nltk


def remove_mentions(text, user_mentions):
    """ Remove mentions from a tweet (catches some missed by NLTK tokenizer)"""
    new_text = text
    usernames = [mention['username'] for mention in user_mentions]
    for username in usernames:
        new_text = re.sub(r'@+'+username, '@USER', new_text, flags=re.IGNORECASE)
    return new_text


def remove_specified_urls(text, urls):
    """ Remove URLs from a text (with URLs provided, like in Tweet objects) """
    new_text = text
    urls = [entity['url'] for entity in urls]
    for url in urls:
        #new_text = new_text.replace(url, '<URL>')
        new_text = new_text.replace(url, '')
    return new_text


def remove_urls(text):
    """ Remove URLs from a text """
    return re.sub(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', '', text)


def remove_nonlatin(text):
    """ Remove non-Latin characters from a text """
    return re.sub(u'[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF]', '', text)


def process_tweet(text, user_mentions, urls, tokenizer):
    """ Process tweet text when user mentions and urls are available """
    new_text = text
    if isinstance(user_mentions, list):
        new_text = remove_mentions(new_text, user_mentions)
    if isinstance(urls, list):
        new_text = remove_specified_urls(new_text, urls)
    return process_tweet_text(text, tokenizer)


def process_tweet_text(text, tokenizer):
    """ Process tweet text (if user mentions and urls are not available) """
    new_text = text
    new_text = remove_urls(new_text)
    new_text = remove_nonlatin(new_text)
    tokens = tokenizer.tokenize(new_text)
    n_tokens = len(tokens)
    new_text = ' '.join(tokens)
    new_text = new_text.replace('RT : ', '')
    new_text = new_text.replace('_', '') # specific to Rieger+2021 dataset issues
    new_text = re.sub('http|https', '', new_text) # specific to Rieger+2021 dataset issues
    return (new_text.lower(), n_tokens)


def tokenize_lowercase(inp):
    """ Tokenize and lowercase text """
    tokens = nltk.word_tokenize(str(inp))
    n_tokens = len(tokens)
    return (' '.join(tokens).lower(), n_tokens)


def process_reddit(inp):
    """ Remove HTML entities, tokenize and lowercase text from Reddit comments
        gathered through PushShift """
    return tokenize_lowercase(remove_urls(html.unescape(str(inp))))


class MLStripper(HTMLParser):

    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def remove_special(text):
    text = text.replace('>', '')
    text = re.sub(r'\d{7,}', '', text)
    text = re.sub(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', '', text) # Remove URLs
    return text


def process_4chan(text):
    """ Preprocess 4chan data """
    # Remove HTML
    text = strip_tags(str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters
    text = remove_special(text)
    return tokenize_lowercase(text)


def process_article(inp):
    text = str(inp).replace('.', '. ')
    return tokenize_lowercase(text)


def process_chat(text, tokenizer, remove_list=None):
    """ Process Discord chat from a random sample dataset 
        Args:
            text: input text as a string
            tokenizer: the tokenizer to use (expecting NLTK TweetTokenizer)
            remove_list: optional list of terms to remove
    """
    if ': ' in text:
        res =  text.split(': ')[1]
    else:
        res = text
    res = re.sub(r'<@!?\d+>', '', res)
    res = remove_urls(res)
    tokens = tokenizer.tokenize(res)
    if remove_list is not None:
        tokens = [tok for tok in tokenizer.tokenize(res) if tok not in remove_list]
    n_tokens = len(tokens)
    return (' '.join(tokens).lower(), n_tokens)


def load_now(fpath):
    """ Load NOW article files into a pandas DataFrame """
    fname = os.path.basename(fpath)
    m = re.search(r'\d\d-\d\d', fname)
    if m is None:
        m = re.search(r'\d\d_\d\d', fname)
        date_str = m.group()
        date = datetime.strptime(date_str, '%y_%m')
    else:
        date_str = m.group()
        date = datetime.strptime(date_str, '%y-%m')
    year = date.year
    with open(fpath) as f:
        articles = f.read().splitlines()
    return pd.DataFrame({'article': articles, 'year': year})


def process_now(inp):
    """ Preprocess NOW articles """
    text = re.sub(r'@@\d+ ', '', inp)
    text = re.sub(r'<\w+>', '', text)
    text = text.replace('@ @ @ @ @ @ @ @ @ @ ', '')
    return (text.lower(), len(text.split()))


def process_rieger2021(text):
    """ Preprocess Rieger+ 2021 4chan, 8chan, t_D data """
    # TODO just make this a preprocess classmethod for Rieger2021 like is done with Stormfront
    # Remove special characters
    text = remove_special(str(text))
    # Tokenize
    return tokenize_lowercase(text)
