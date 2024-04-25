from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, BatchEncoding

class WikipediaDataset:

    def __init__(self, csv_path: str, tokenizer: T5Tokenizer, n_articles: Optional[int] = None) -> None:
        self.csv_path: str = csv_path
        self.tokenizer: T5Tokenizer = tokenizer
        self.n_articles: Optional[int] = n_articles

    @lru_cache(maxsize=5)
    def __call__(
        self, 
        split_ratios: Tuple[float, float] = None
    ) -> DatasetDict:
        
        table: pd.DataFrame = pd.read_csv(
            self.csv_path, 
            names=['category', 'topic', 'summary', 'text', 'page_id', 'url'],
            header=0,
            dtype=str,
            nrows=self.n_articles,
        ).dropna(subset=['topic','summary','text'], how='any')
        # convert `pd.DataFrame` to `datasets.Dataset`
        dataset: Dataset = Dataset.from_pandas(df=table)

        # TODO: improve data loader for inference (no splits, no tokenizers)
        if split_ratios is not None:
            if sum(split_ratios) != 1:
                raise ValueError('sum of split_ratios must be 1')
            else:
                self.split_ratios: Tuple[float, float] = split_ratios
                # split the dataset into train, validation, and test sets
                train_val: DatasetDict = dataset.train_test_split(
                    train_size=split_ratios[0], 
                    test_size=split_ratios[1], 
                    seed=42,
                )
                train_dataset: Dataset = train_val['train']
                val_dataset: Dataset = train_val['test']
                datadict: DatasetDict = DatasetDict({'train': train_dataset, 'val': val_dataset})
        else:
            datadict: DatasetDict = DatasetDict({'all': dataset})

        # TODO: only get first 512 tokens
        return datadict.map(
            function=self.__batch_preprocess, 
            batched=True, 
            batch_size=1024,
            num_proc=4,
        )

    def __batch_preprocess(self, batch: Dataset) -> Dataset:
        batch['text'] = ['summarize: ' + doc for doc in batch['text']]
        batch['input_ids'], batch['attention_mask'] = self.__tokenize(batch['text'], max_length=2560)
        batch['labels'], _ = self.__tokenize(batch['summary'], max_length=256)
        return batch
    
    def __tokenize(self, batch: Dataset, max_length: int) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[str]]]]:
        encoding: BatchEncoding = self.tokenizer(text_target=batch, max_length=max_length, truncation=True)
        input_ids: Dict[str, List[List[str]]] = encoding['input_ids']
        attention_mask: Dict[str, List[List[str]]] = encoding['attention_mask']
        return input_ids, attention_mask
    
if __name__ == '__main__':
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large', model_max_length=5120)
    wiki = WikipediaDataset(csv_path='datasets/WikiData.csv', tokenizer=tokenizer)
    dataset = wiki(split_ratios=(0.8, 0.1, 0.1))


