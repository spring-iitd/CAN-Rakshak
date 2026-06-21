import abc


class DataAugmentor(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def augment(self, input_path, output_path):
        """Generate a synthetic dataset from input_path and write it to output_path."""
        raise NotImplementedError
