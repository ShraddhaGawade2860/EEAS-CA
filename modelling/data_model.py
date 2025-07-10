class Data:
    def __init__(self, X, df, label):
        self.X = X
        self.y = df[label]
        self.X_train = X
        self.X_test = X