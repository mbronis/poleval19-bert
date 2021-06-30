from datasets import load_dataset

from src.utils import Logger


class DataLoader:

    def __init__(self):
        self.dataset_name = 'cdt'
        self.logger = Logger('io')

    def load_raw(self):
        self.logger.log('Loading raw dataset')
        return load_dataset(self.dataset_name)
