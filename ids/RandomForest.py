from common_imports import accuracy_score, joblib
from sklearn.ensemble import RandomForestClassifier
from ids.base import IDS

class RandomForest(IDS):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=4)

    def train(self, **kwargs):
        super().train()
        self.rf.fit(self.X, self.Y)

    def test(self, **kwargs):
        Y_pred = self.predict(self.X)
        return Y_pred, self.Y

    def save(self, path):
        joblib.dump(self.rf, path)

    def predict(self, X_test):
        rf_preds = self.rf.predict(X_test)
        return rf_preds

    def load(self, path):
        self.rf = joblib.load(path)

    def extract_features(self):
        pass

