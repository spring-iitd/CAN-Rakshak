from common_imports import abc


class BaseSplitter:
    def __init__(self, input_dir, feature_extractor="PixNet", **kwargs):  # PixNet: work in progress, will be released upon publication
        self.input_dir = input_dir
        self.feature_extractor = feature_extractor
        self.kwargs = kwargs
        
    @abc.abstractmethod
    def split(self):
        raise NotImplementedError("Subclasses must implement this method.")

