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

SCRIPTPARFOLDER = os.path.dirname(os.path.dirname(__file__))
os.chdir(SCRIPTPARFOLDER)

from train_predict.train import import_preproc_tweets


#%%
# set up logging

logging.basicConfig(
    filename = r'./log/sentiment_analysis.log',
    format = "[%(asctime)s] (%(name)s) %(message)s"
)
logger = logging.getLogger(name = 'Predicter script')
logger.setLevel(logging.INFO)

#%%
# load model

def load_model(load_path):
    logger.info('Loading model.. ')
    with open(load_path, 'rb') as loadfile:
        full_model_spec = pickle.load(loadfile)

    model = full_model_spec['model']
    model_params = full_model_spec['model_params']
    X_train = full_model_spec['X_train']
    y_train = full_model_spec['y_train']

    logger.info('..loaded.')
    return model, model_params, X_train, y_train

#%%
# predict based on fitted model

def predict_model(model, tweets):

    logger.info('Predicting..')
    X = tweets['radicals'] + ' ' + tweets['query_used']

    y_pred = model.predict(X)

    logger.info('..predicted.')
    return y_pred

#%%
# save prediction
def save_prediction(
    save_path,
    tweets,
    y_pred,
    full_tweet_db = True,
):  
    logger.info('Setting up output database...')
    if full_tweet_db:
        prediction = tweets.copy()
        prediction['sentiment_predict'] = y_pred
    else:
        prediction = pd.DataFrame(y_pred, index = tweets.index)
        prediction.columns = ['sentiment_predict']
    logger.info('..set up.')

    logger.info(f"Saving to '{os.path.abspath(save_path)}'..")
    prediction.to_csv(save_path)
    logger.info('..saved.')

    return prediction

#%%
# go
if __name__ == '__main__':

    logger.info('Start.')

    tweets = import_preproc_tweets(
        tweets_db_path = r'../data/Subm3Classes.csv', 
        sample_size = None, 
        stopwords_db_path = r'../data/stopwords_alopes.txt'
    )

    model, model_params, X_train, y_train = load_model(r'model.pickle')

    y_pred = predict_model(model, tweets)

    prediction = save_prediction(
        save_path = r'./prediction.csv',
        tweets = tweets,
        y_pred = y_pred,
        full_tweet_db = False,
    )

    logger.info('Done with script.')


# %%
