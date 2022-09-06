""" Utility functions for processing data, etc 

    @author Michael Miller Yoder
    @year 2022
"""

import re
from html.parser import HTMLParser

import nltk


def remove_mentions(text, user_mentions):
    """ Remove mentions from a tweet (catches some missed by NLTK tokenizer)"""
    new_text = text
    usernames = [mention['username'] for mention in user_mentions]
    for username in usernames:
        new_text = re.sub(r'@+'+username, '@USER', new_text, flags=re.IGNORECASE)
    return new_text


def remove_urls(text, urls):
    """ Remove URLs from a text """
    new_text = text
    urls = [entity['url'] for entity in urls]
    for url in urls:
        new_text = new_text.replace(url, '<URL>')
    return new_text


def process_tweet(text, user_mentions, urls, tokenizer):
    new_text = text
    if isinstance(user_mentions, list):
        new_text = remove_mentions(new_text, user_mentions)
    if isinstance(urls, list):
        new_text = remove_urls(new_text, urls)
    new_text = ' '.join(tokenizer.tokenize(new_text))
    return new_text.lower()


def tokenize_lowercase(inp):
    """ Tokenize and lowercase text """
    return ' '.join(nltk.word_tokenize(str(inp))).lower()


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
