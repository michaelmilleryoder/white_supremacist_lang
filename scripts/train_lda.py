""" Train LDA on a corpus. """

import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def main():

    # Settings
    name = 'white_supremacist_fulltext'
    input_path = f'../tmp/{name}_train_corpus.pkl'
    vec_outpath = f'../models/{name}_countvectorizer.pkl'
    n_topics = 30
    model_outpath = f'../models/{name}_lda{n_topics}.pkl'
    max_iter = 100

    # Load data
    print("Loading data...")
    ws_data = pd.read_pickle(input_path)

    # Process text
    print("Processing text...")
    vectorizer = CountVectorizer(min_df=1, stop_words='english')
    bow = vectorizer.fit_transform(ws_data.text)
    with open(vec_outpath, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Train LDA model
    print("Training LDA model...")
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, n_jobs=-1, verbose=2)
    lda.fit(bow)

    # Save LDA model
    with open(model_outpath, 'wb') as f:
        pickle.dump(lda, f)
    print("LDA model saved to ", model_outpath)


if __name__ == '__main__':
    main()
