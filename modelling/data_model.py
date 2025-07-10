import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame, label: str):
        self.label = label
        self.X = X
        self.y = df[label]
        self.X_train = X
        self.X_test = X
