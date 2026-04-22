from common_imports import (
    os, np, logging, joblib,
    accuracy_score, StandardScaler,
    Dense, Input, Sequential, SparseCategoricalCrossentropy, EarlyStopping,
)
from ids.base import IDS
import absl.logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("absl").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
class MLP(IDS):
    def __init__(self):
        self.mlp = Sequential()
        self.mlp.add(Input(shape = (4,)))
        self.mlp.add(Dense(128, activation = 'relu'))
        self.mlp.add(Dense(128, activation = 'relu'))
        self.mlp.add(Dense(4, activation='softmax'))

    def train(self, cfg=None, **kwargs):
        cfg = cfg or {}
        # super().train(X_train, Y_train)
        super().train()
        X_train = np.array(self.X).astype("float32")
        Y_train = np.array(self.Y).astype("int32")

        self.mlp.compile(optimizer='adam',
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

        self.es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

        self.mlp_hist = self.mlp.fit(X_train, Y_train, epochs=cfg.get('epochs', 10), validation_split=0.2, callbacks = [self.es], batch_size = 8192)

    def test(self, **kwargs):
        X_test = np.array(self.X).astype("float32")
        Y_test = np.array(self.Y).astype("int32")
        Y_pred = self.predict(X_test)
        return Y_pred, Y_test

    def save(self, path):
        joblib.dump(self.mlp, path)

    def predict(self, X_test):
        # super().predict()
        # X_test = np.array(self.X).astype("float32")
        return self.mlp.predict(X_test, batch_size=8192).argmax(axis=1)
        # return self.mlp.predict(X_test).argmax(axis=1)

    def load(self, path):
        self.mlp = joblib.load(path)

    def extract_features(self):
        pass
    

