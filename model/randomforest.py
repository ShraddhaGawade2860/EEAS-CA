# Remove this line if it's not needed
# from model.base import BaseModel  ← this line causes the error if base.py doesn't exist

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib

class RandomForest:
    def __init__(self, model_name: str, X: np.ndarray, y: np.ndarray) -> None:
        self.name = model_name
        self.X = X
        self.y = y
        self.clf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        self.predictions = None

    def train_data(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict_data(self, X_test):
        self.predictions = self.clf.predict(X_test)
        return self.predictions

    def print_results(self, y_true):
        print(f"\n[RESULTS for {self.name}]")
        print(classification_report(y_true, self.predictions))
        self.plot_confusion_matrix(y_true)

    def plot_confusion_matrix(self, y_true):
        cm = confusion_matrix(y_true, self.predictions, labels=np.unique(y_true))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {self.name}')
        plt.show()

    def save_model(self, filepath):
        joblib.dump(self.clf, filepath)

    def load_model(self, filepath):
        self.clf = joblib.load(filepath)
