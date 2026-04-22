from ids.base import IDS
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
class DecisionTree(IDS):
    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth = 4)

    def train(self, **kwargs):
        super().train()
        self.dt.fit(self.X, self.Y)

    def test(self, **kwargs):
        Y_pred = self.predict(self.X)
        return Y_pred, self.Y

    def save(self, path):
        joblib.dump(self.dt, path)
    
    def predict(self, X_test):
        dt_preds = self.dt.predict(X_test)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)

    def extract_features(self):
        pass

