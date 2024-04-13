import re
import string
from functools import lru_cache
from typing import List, Tuple, Dict
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, BatchEncoding

class WikipediaDataset:

    def __init__(
        self, 
        csv_path: str,
        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-large")
    ) -> None:
        self.csv_path: str = csv_path
        self.tokenizer: T5Tokenizer = tokenizer

    @lru_cache(maxsize=5)
    def __call__(
        self, 
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> DatasetDict:
        
        self.split_ratios: Tuple[float, float, float] = split_ratios
        table: pd.DataFrame = pd.read_csv(
            self.csv_path, 
            names=['category', 'topic', 'summary', 'text', 'page_id', 'url'],
            header=0,
            dtype=str,
            # TODO: drop nrows
            nrows=1000,
        ).fillna(value='')
        # convert `pd.DataFrame` to `datasets.Dataset`
        dataset: Dataset = Dataset.from_pandas(df=table)
        # split the dataset into train, validation, and test sets
        train_valtest: DatasetDict = dataset.train_test_split(
            train_size=split_ratios[0], 
            test_size=split_ratios[1] + split_ratios[2], 
            seed=42,
        )
        train_dataset: Dataset = train_valtest['train']
        valtest_dataset: Dataset = train_valtest['test']
        val_test: DatasetDict = valtest_dataset.train_test_split(
            train_size=split_ratios[1] / sum(split_ratios[1:]), 
            seed=42
        )
        val_dataset: Dataset = val_test['train']
        test_dataset: Dataset = val_test['test']
        datadict: DatasetDict = DatasetDict({
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
        })
        datadict = datadict.map(
            function=self.__batch_preprocess, 
            batched=True, 
            batch_size=1024,
            num_proc=6,
        )
        return datadict

    def __batch_preprocess(self, batch: Dataset) -> Dataset:
        field: str
        for field in ['summary', 'text']:
            batch[field] = [self.__preprocess_text(doc) for doc in batch[field]]
        batch['text'] = ['summarize: ' + doc for doc in batch['text']]
        batch['input_ids'], batch['attention_mask'] = self.__tokenize(batch['text'])
        batch['labels'], _ = self.__tokenize(batch['summary'])
        return batch
    
    def __tokenize(self, batch: Dataset) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[str]]]]:
        encoding: BatchEncoding = self.tokenizer(text_target=batch, max_length=10240, truncation=True)
        input_ids: Dict[str, List[List[str]]] = encoding['input_ids']
        attention_mask: Dict[str, List[List[str]]] = encoding['attention_mask']
        return input_ids, attention_mask

    @staticmethod
    def __preprocess_text(raw_text: str) -> str:
        result: str = raw_text.lower()
        result: str = re.sub(pattern=r'\{\\displaystyle [^\}]+\}', repl='', string=result)
        result: str = re.sub(pattern=r'\{[^}]*\}', repl='', string=result)
        result: str = re.sub(pattern=r'\\[a-zA-Z]+\{[^}]*\}', repl='', string=result)
        result: str = re.sub(pattern=r'\\[a-zA-Z]+', repl='', string=result)
        result: str = re.sub(pattern='\u202f' ,repl=' ', string=result)
        result: str = re.sub(pattern='\xa0' ,repl=' ', string=result)
        result: str = re.sub(pattern=r'\s[\W]+\s', repl=r' ', string=result)
        result: str = re.sub(pattern=r'(\w)([^\w\s])', repl=r'\1 \2', string=result)
        result: str = re.sub(pattern=r'([^\w\s])(\w)', repl=r'\1 \2', string=result)
        result: str = re.sub(pattern=r'[\n\t\s]+', repl=' ', string=result)
        return result.strip(string.punctuation + ' ')
    
    
