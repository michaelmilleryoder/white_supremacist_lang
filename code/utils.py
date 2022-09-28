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
    new_text = text
    if isinstance(user_mentions, list):
        new_text = remove_mentions(new_text, user_mentions)
    if isinstance(urls, list):
        new_text = remove_specified_urls(new_text, urls)
    new_text = ' '.join(tokenizer.tokenize(new_text))
    return new_text.lower()


def process_tweet_text(text, tokenizer):
    """ Process tweet text (if user mentions and urls are not available) """
    new_text = text
    new_text = remove_urls(new_text)
    new_text = remove_nonlatin(new_text)
    new_text = ' '.join(tokenizer.tokenize(new_text))
    new_text = new_text.replace('RT : ', '')
    new_text = new_text.replace('_', '') # specific to Rieger+2021 dataset issues
    new_text = re.sub('http|https', '', new_text) # specific to Rieger+2021 dataset issues
    return new_text.lower()


def tokenize_lowercase(inp):
    """ Tokenize and lowercase text """
    return ' '.join(nltk.word_tokenize(str(inp))).lower()


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
    # Tokenize
    text = ' '.join(nltk.word_tokenize(str(text))).lower()
    return text


def process_article(inp):
    text = ' '.join(nltk.word_tokenize(str(inp.replace('.', '. ')))).lower()
    return text


def process_chat(text, tokenizer):
    """ Process Discord chat from a random sample dataset 
        Args:
            text: input text as a string
            tokenizer: the tokenizer to use (expecting NLTK TweetTokenizer)
    """
    if ': ' in text:
        res =  text.split(': ')[1]
    else:
        res = text
    return ' '.join(tokenizer.tokenize(res)).lower()

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
    return text.lower()
