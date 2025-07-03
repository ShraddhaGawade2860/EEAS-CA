from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_tfidf_embd(df):
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['interaction_content']).toarray()
    return X, df
