"""
    Scrape Medium articles by tag

    @author Michael Miller Yoder
    @year 2022
"""


import requests
import pdb
from calendar import Calendar
from datetime import date, timedelta
import dateutil.parser as dparser
import time

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():

    # Settings
    tag = 'white-supremacy'
    url_outpath = f'../../data/antiracist/medium/{tag}_urls.txt'
    article_outpath = f'../../data/antiracist/medium/{tag}_articles.jsonl'
    start_year = 2010 # 2010 in white supremacist data
    start_month = 1
    start_url_idx = 1113 # if starting partway through URLs
    override = True

    # Run
    #print("Getting article URLs...")
    #urls = get_urls(tag, url_outpath, start_year, start_month, override)
    urls = load_urls(url_outpath)
    print("Getting article texts...")
    get_articles(urls, start_url_idx, article_outpath)


def load_urls(path):
    """ Load URLs from file to scrape """
    with open(path) as f:
        return f.read().splitlines()


def get_articles(urls, start_url_idx, article_outpath):
    """ Get and save out article text from URLs """
    outlines = []
    for url in tqdm(urls[start_url_idx:], ncols=100):
        try:
            response = requests.get(url)
            if response.status_code in [403, 404, 410]:
                tqdm.write(f'{response.status_code} error, skipping {url}')
                continue
            elif response.status_code in [500, 503, 520, 524]:
                sleep_time = 30
                tqdm.write(f'{response.status_code} error, waiting {sleep_time} seconds and then trying again')
                time.sleep(sleep_time)
            elif response.status_code == 200:
                soup = BeautifulSoup(response.content, features='html.parser')
                title = ''
                title_search = soup.find('h1', class_='pw-post-title') 
                if title_search is not None:
                    title = title_search.text
                date = None
                date_search = soup.find('p', class_='pw-published-date')
                if dat_search is not None:
                    date_str = date_search.text
                    date = dparser.parse(date_str).strftime('%Y-%m-%d')
                paras = [p.text for p in soup.find_all('p', class_='pw-post-body-paragraph')]
                outlines.append({'url': url, 'date': date, 'title': title, 'text': '\n'.join([title] + paras).strip()})
                time.sleep(0.5)
            else: 
                pdb.set_trace()
        except Exception as e:
            tqdm.write(f'{str(e)}\n\t for {url}')
            pd.DataFrame(outlines).to_json(article_outpath, orient='records', lines=True)
            continue

    pd.DataFrame(outlines).to_json(article_outpath, orient='records', lines=True)
    print(f"Wrote article texts to {article_outpath}")


def extract_articles(url):
    """ Extract article mentions from a Medium page URL """
    response = requests.get(url, allow_redirects=False)
    if response.status_code != 200:
        tqdm.write(f"No articles found for {url}")
        return (None, None)
    soup = BeautifulSoup(response.content, features='html.parser')
    articles = soup.find_all("div", class_="postArticle postArticle--short js-postArticle js-trackPostPresentation js-trackPostScrolls")
    return soup, articles


def extract_urls(articles):
    """ Extract article URLs from a list of BeautifulSoup Tag elements that should have them """
    article_urls = []
    for article in articles:
        links = article.find_all("a")
        article_urls.append(links[-1]['href'].split('?')[0])
        #if len(links) >= 4:
        #    if '?' in links[3]['href']:
        #        article_urls.append(links[3]['href'].split('?')[0])
        #else:
        #    pdb.set_trace()
    return article_urls


def get_urls(tag, url_outpath, start_year, start_month, override):
    """ Search for URLs of articles that match the tag 
        Returns and saves a list of the URLs
    """
    url_base = 'https://medium.com/tag/{}/archive/{}/{:02d}'
    all_urls = []
    c = Calendar()
    with open(url_outpath, 'w' if override else 'a') as f:
        for year in tqdm(range(start_year, 2023), ncols=100):
            for month in range(start_month, 13):
                month_article_urls = []
                month_url = url_base.format(tag, year, month)
                soup, articles = extract_articles(month_url)
                if soup is None:
                    continue
        
                if len(articles) < 10: # Fewer than 10 means don't have to search by day
                    tqdm.write(f'{len(articles)} articles found at {month_url}')
                    month_article_urls += extract_urls(articles)
                else:
                    calendar_mention = soup.find('p', class_='u-marginBottom40')
                    if calendar_mention.text == '': # exactly 10 articles
                        tqdm.write(f'{len(articles)} articles found at {month_url}')
                        month_article_urls += extract_urls(articles)
                    else: # Try calendar date URLs
                        for date in [d for d in c.itermonthdates(year, month) if d.month == month]:
                            day_url = month_url + '/{:02d}'.format(date.day)
                            soup, articles = extract_articles(day_url)
                            if soup is None:
                                continue
                            tqdm.write(f'{len(articles)} articles found at {day_url}')
                            month_article_urls += extract_urls(articles)
                all_urls += month_article_urls
                f.write('\n'.join(month_article_urls) + '\n')

    print(f"Wrote URLs to {url_outpath}")
    return all_urls


if __name__ == '__main__':
    main()
