from __future__ import annotations

import re
from typing import Dict, List

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

from src.data_types import Tweet


class Preprocessor:

    def __call__(self, dataset_dict: DatasetDict) -> DatasetDict:
        return dataset_dict.map(Preprocessor._preprocess)

    @staticmethod
    def _preprocess(dataset: Dataset) -> Dict[str, List[Tweet]]:
        tweets_preprocessed = {k: v for k, v in dataset.items()}
        tweets_preprocessed['preproc'] = Preprocessor._base_cleanup(dataset['sentence'])
        return tweets_preprocessed

    @staticmethod
    def _base_cleanup(tweet: Tweet) -> Tweet:
        """Clean-up tweet text"""
        tweet = tweet.strip()
        tweet = re.sub(r'@anonymized_account', '', tweet)
        tweet = re.sub(r'[^\w\s\.?!%,]', '', tweet)
        tweet = re.sub(r' +', ' ', tweet)
        tweet = tweet.strip()

        return tweet
