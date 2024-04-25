from typing import List, Tuple, Dict, Optional
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, BatchEncoding

class WikipediaDataset:

    def __init__(self, csv_path: str) -> None:
        self.csv_path: str = csv_path
    
    def get_raw_dataset(self, n_articles: Optional[int] = None) -> Dataset:
        # read csv file
        table: pd.DataFrame = pd.read_csv(
            self.csv_path, 
            names=['category', 'topic', 'summary', 'text', 'page_id', 'url'],
            header=0,
            dtype=str,
            nrows=n_articles,
        ).dropna(subset=['topic','summary','text'], how='any')
        # convert `pd.DataFrame` to `datasets.Dataset`
        return Dataset.from_pandas(df=table)

    def for_training(
        self, 
        tokenizer: T5Tokenizer,
        split_ratios: Tuple[float, float] = (0.8, 0.2),
        n_articles: Optional[int] = None,
    ) -> DatasetDict:
        
        if sum(split_ratios) != 1:
            raise ValueError('sum of split_ratios must be 1')

        # get the raw dataset
        dataset: Dataset = self.get_raw_dataset(n_articles)
        # split the dataset into train, validation, and test sets
        train_val: DatasetDict = dataset.train_test_split(
            train_size=split_ratios[0], 
            test_size=split_ratios[1], 
            seed=42,
        )
        train_dataset: Dataset = train_val['train']
        val_dataset: Dataset = train_val['test']
        datadict: DatasetDict = DatasetDict({'train': train_dataset, 'val': val_dataset})

        return datadict.map(
            function=lambda batch: self.__batch_preprocess(tokenizer, batch), 
            batched=True, 
            batch_size=1024,
            num_proc=4,
        )
    
    def for_inference(self, max_article_length: int = 512) -> DatasetDict:
        # get the raw dataset
        dataset: Dataset = self.get_raw_dataset(None)
        return DatasetDict({'all': dataset}).map(
            function=lambda batch: WikipediaDataset.__batch_truncate(batch, max_article_length),
            batched=True,
            batch_size=1024,
            num_proc=4,
        )

    @staticmethod
    def __batch_truncate(batch: Dataset, max_article_length: int) -> Dataset:
        batch['text'] = ['summarize: ' + ' '.join(text.split()[:max_article_length]) for text in batch['text']]
        return batch

    @staticmethod
    def __batch_preprocess(tokenizer: T5Tokenizer, batch: Dataset) -> Dataset:
        batch['text'] = ['summarize: ' + text.replace(summary, '') for text, summary in zip(batch['text'], batch['summary'])]
        batch['input_ids'], batch['attention_mask'] = WikipediaDataset.__tokenize(tokenizer, batch['text'], max_length=1024)
        batch['labels'], _ = WikipediaDataset.__tokenize(tokenizer, batch['summary'], max_length=256)
        return batch

    @staticmethod    
    def __tokenize(
        tokenizer: T5Tokenizer, 
        batch: Dataset, 
        max_length: int
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[str]]]]:
        encoding: BatchEncoding = tokenizer(text_target=batch, max_length=max_length, truncation=True)
        input_ids: Dict[str, List[List[str]]] = encoding['input_ids']
        attention_mask: Dict[str, List[List[str]]] = encoding['attention_mask']
        return input_ids, attention_mask
    
if __name__ == '__main__':
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large', model_max_length=5120)
    wiki = WikipediaDataset(csv_path='datasets/WikiData.csv')
    dataset_training: DatasetDict = wiki.for_training(tokenizer, n_articles=1000)
    dataset_inference: DatasetDict = wiki.for_inference(max_article_length=512)


