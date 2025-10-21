from abc import ABC, abstractmethod

class IDS(ABC):
    @abstractmethod
    def train(self, X_train=None, Y_train=None, **kwargs):
        """
        Train the model using the provided training dataset.

        :param X_train: Features of the training dataset.
        :param Y_train: Labels of the training dataset.
        :param kwargs: Additional keyword arguments for the training process.
        """
        pass

    @abstractmethod
    def test(self, X_test=None, Y_test=None, **kwargs):
        """
        Test the model using the provided test dataset.

        :param X_test: Features of the test dataset.
        :param Y_test: Labels of the test dataset.
        :param kwargs: Additional keyword arguments for the testing process.
        """
        pass

    @abstractmethod
    def predict(self,X_test=None, **kwargs):
        """
        Predict the output using the model for the given input features.

        :param X_test: Features for which predictions are to be made.
        :param kwargs: Additional keyword arguments for the prediction process.
        :return: The predicted output.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save the trained model to the specified path.

        :param path: The file path where the model will be saved.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load a trained model from the specified path.

        :param path: The file path from which the model will be loaded.
        """
        pass

