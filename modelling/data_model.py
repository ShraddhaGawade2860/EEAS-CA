import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame, label: str):
        self.label = label
        self.X = X
        self.y = df[label].to_numpy()

        # Filter out missing/empty labels
        valid = ~pd.isna(self.y) & (self.y != '')
        self.X = self.X[valid]
        self.y = self.y[valid]

        # Filter classes with 3 samples
        y_series = pd.Series(self.y)
        good_classes = y_series.value_counts()[y_series.value_counts() >= 3].index
        keep = y_series.isin(good_classes)

        self.X = self.X[keep]
        self.y = self.y[keep]

        if len(np.unique(self.y)) < 2:
            self.skip = True
            return
        else:
            self.skip = False

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_all_data(self):
        return self.X, self.y

    def should_skip(self):
        return self.skip