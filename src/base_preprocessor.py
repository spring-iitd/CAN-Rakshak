import os
from abc import ABC, abstractmethod
import shutil   
from config import *
from utilities import *

class DataPreprocessor(ABC):
    # Abstract base class for data preprocessors

    def run(self, dataset_path, **kwargs):
        """
        This method moves all files from dataset_path to a new folder "original_dataset"
        and then calls the preprocess_dataset method to preprocess the data and save it
        to "modified_dataset" folder.
        """
        files = os.listdir(dataset_path)
        for file in files : 
            full_file_path = os.path.join(dataset_path,file)
            if os.path.isdir(full_file_path) or file.endswith('.py'):
                continue
            self._move_to_original(full_file_path)
        orig_file_path = os.path.join(dataset_path,"original_dataset")
        modified_file_path = self._get_modified_dataset_path(dataset_path)
        self.preprocess_dataset(orig_file_path, modified_file_path, **kwargs)

    def _move_to_original(self, file_path):
        """Moves the given file to the original_dataset folder. Creates the folder if it doesn't exist."""
        dir_path = os.path.dirname(file_path)
        orig_dir_path = os.path.join(dir_path, "original_dataset")
        os.makedirs(orig_dir_path, exist_ok=True)
        filename = os.path.basename(file_path)
        dest_path = os.path.join(orig_dir_path, filename)
        shutil.move(file_path, dest_path)  

    def _get_modified_dataset_path(self, dataset_path):
        """Returns the path to the modified_dataset folder. Creates the folder if it doesn't exist."""
        mod_dir_path = os.path.join(dataset_path, "modified_dataset")
        os.makedirs(mod_dir_path, exist_ok=True)

        return mod_dir_path

    @abstractmethod
    def preprocess_dataset(self, orig_file_path, modified_file_path, **kwargs):
        """
        User should read from "orig_file_path" and save to "modified_file_path" folder.
        1. Read the dataset from orig_file_path
        2. Preprocess the dataset as per requirement
        3. Save the preprocessed dataset to modified_file_path
        4. Additional parameters can be passed using kwargs
        5. This method should be overridden by the user.
        """
        pass
