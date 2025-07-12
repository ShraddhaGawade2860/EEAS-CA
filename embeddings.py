import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]

    num_docs = len(df)
    # Dynamically adjust min_df and max_df for small datasets
    if num_docs < 10:
        min_df = 1
        max_df = 1.0
    elif num_docs < 50:
        min_df = 2
        max_df = 0.95
    else:
        min_df = 4
        max_df = 0.90

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=min_df, max_df=max_df)
    X = tfidfconverter.fit_transform(data).toarray()
    return X


def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

