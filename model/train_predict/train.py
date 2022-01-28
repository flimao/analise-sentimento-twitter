#!/usr/bin/env python3
#-*- coding: utf-8 -*-

#%% 
# installs
# pip install git+ssh://git@github.com/flimao/mltoolkit

#%%
# imports

import os
import pickle
import pandas as pd
import numpy as np
import logging
from mltoolkit import NLP
import spacy

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from mltoolkit.NLP import W2VTransformer

SCRIPTPARFOLDER = os.path.dirname(os.path.dirname(__file__))
os.chdir(SCRIPTPARFOLDER)

# python -m spacy download pt_core_news_lg
# python -m spacy download pt_core_news_md
# python -m spacy download pt_core_news_sm
nlp = spacy.load("pt_core_news_lg")

#%%
# set up logging

logging.basicConfig(
    filename = r'./log/sentiment_analysis.log',
    format = "[%(asctime)s] (%(name)s) %(message)s"
)
logger = logging.getLogger(name = 'Trainer script')
logger.setLevel(logging.INFO)

#%% 
# import train tweets

def import_preproc_tweets(
    tweets_db_path, 
    sample_size = None, 
    stopwords_db_path = r'./stopwords_alopes.txt'
):

    logger.info('Reading database.. ')
    tweets_raw = pd.read_csv(
        tweets_db_path,
    )
    logger.info('read.')

    logger.info('Pre-processing database.. ')
    def mudar_tipos(df):
        df = df.copy()

        df['id'] = df['id'].astype('string')
        df['tweet_date'] = pd.to_datetime(df['tweet_date'])
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype('category')

        return df

    def remover_duplicatas(df):
        df = df.copy()

        df = df.drop_duplicates(subset = 'id')

        return df

    def setar_index(df):
        df = df.copy()

        df = df.set_index('id')

        return df

    tweets_full = (tweets_raw
        .pipe(mudar_tipos)
        .pipe(remover_duplicatas)
        .pipe(setar_index)
    )
    logger.info('pre-processed.')

    # stopwords
    logger.info('Building text pre-processing function.. ')
    if stopwords_db_path is None:
        stopwords_set = set()
    else:
        with open(stopwords_db_path, encoding = 'utf8') as stopword_list:
            lst = stopword_list.read().splitlines()

        stopwords_set = set([ stopword.strip() for stopword in lst ])

        # don't remove negative words
        remover_stopwords = {
            'não', 
        }

        stopwords_set -= remover_stopwords

    # build text preproc function
    preprocessing_full = lambda s: NLP.preprocessing(s, preproc_funs_args = [
        NLP.remove_links,
        NLP.remove_hashtags,
        NLP.remove_mentions,
        NLP.remove_numbers,
        NLP.remove_special_caract,
        NLP.lowercase,
        (NLP.tokenize_remove_stopwords_get_radicals_spacy, dict(
            nlp = nlp,
            stopword_list = stopwords_set,
        )),
    ])
    logger.info('built.')

    # apply preproc to database (or random sample of it)
    logger.info('Applying pre-processing function to database.. ')
    if sample_size is None:
        radicals = tweets_full['tweet_text'].apply(preprocessing_full)
    else:
        radicals = tweets_full.sample(sample_size)['tweet_text'].apply(preprocessing_full)

    tweets = tweets_full.copy()
    tweets['radicals'] = radicals
    tweets = tweets[tweets.radicals.notna()]
    logger.info(' applied.')

    return tweets
#%%
# run model

def fit_model(
    tweets, 
    transformer,
    estimator,
    model_params,
):
    logger.info('Define data with which to build the ML model.. ')
    X = tweets['radicals'] + ' ' + tweets['query_used']
    y = tweets['sentiment']
    logger.info('defined.')

    logger.info('Building model.. ')
    model = Pipeline(steps = [
        transformer,
        estimator
    ])

    model.set_params(**model_params)
    logger.info('built.')

    logger.info('Fitting model to data.. ')
    model.fit(X, y)
    logger.info('fitted.')

    return model, X, y

def save_model(
    save_path,
    model,
    model_params,
    X, y,
):

    full_model_spec = {
        'model': model,
        'model_params': model_params,
        'X_train': X,
        'y_train': y
    }

    with open(save_path, 'wb') as savefile:
        pickle.dump(full_model_spec, savefile)
    
    return True

#%%
# go!
if __name__ == '__main__':
    
    logger.info('Start.')

    tweets = import_preproc_tweets(
        tweets_db_path = r'../data/Train3Classes.csv',
        sample_size = None,
        stopwords_db_path = r'../data/stopwords_alopes.txt'
    )

    logger.debug('Defining model pipeline components and parameters.. ')
    transformer = ('word2vec', W2VTransformer())
    estimator = ('xgboost', XGBClassifier())

    model_params = {
        'word2vec__vector_combination': 'sum',
        'word2vec__vector_size': 200,
        'xgboost__criterion': 'friedman_mse',
        'xgboost__learning_rate': 0.5,
        'xgboost__max_depth': 4,
        'xgboost__max_features': 'sqrt',
        'xgboost__n_estimators': 150
    }
    logger.debug('defined.')

    model_fitted, X, y = fit_model(
        tweets = tweets,
        transformer = transformer,
        estimator = estimator,
        model_params = model_params,
    )

    save_model(
        save_path = 'model.pickle',
        model = model_fitted,
        model_params = model_params,
        X = X,
        y = y
    )

    logger.info('Done with script.')
    
# %%
