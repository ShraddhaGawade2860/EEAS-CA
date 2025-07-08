import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class RandomForest(BaseModel):
    def __init__(self, model_name: str, X: np.ndarray, y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.name = model_name
        self.X = X
        self.y = y
        self.clf = None
        self.predictions = None

    def train_data(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
        self.clf.fit(X_train, y_train)

    def predict_data(self, X_test):
        self.predictions = self.clf.predict(X_test)
        return self.predictions

    def print_results(self, y_true):
        print(f"\n[RESULTS for {self.name}]")
        print(classification_report(y_true, self.predictions))
        self.plot_confusion_matrix(y_true)

    def save_model(self, filepath):
        joblib.dump(self.clf, filepath)

    def load_model(self, filepath):
        self.clf = joblib.load(filepath)
