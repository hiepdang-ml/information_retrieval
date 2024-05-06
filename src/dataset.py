from typing import List, Tuple, Dict, Optional
import re
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, BatchEncoding

from google.cloud import bigquery


class WikipediaDataset:

    def __init__(self) -> None:
        self.bq_client = bigquery.Client()
    
    def get_raw_dataset(self, n_articles: Optional[int] = None) -> Dataset:
        if n_articles is None:
            limit_command: str = ''
        else:
            limit_command: str = f'LIMIT {int(n_articles)}'
        query = (
            f"""
            SELECT 
                `Category` AS `category`, 
                `Topic` AS `topic`, 
                `Summary` AS `summary`, 
                `Full_Content` AS `text`, 
                `Page_ID` AS `page_id`, 
                `URL` AS `url`
            FROM `analog-button-421413.WikiData.WikiData-Main`
            WHERE TIMESTAMP_TRUNC(_PARTITIONTIME, DAY) = TIMESTAMP("2024-04-26")
                AND `Topic` IS NOT NULL AND `Summary` IS NOT NULL AND `Full_Content` IS NOT NULL
            {limit_command}
            """
        )
        query_job = self.bq_client.query(query)  # make an API request.
        return Dataset.from_pandas(df=query_job.to_dataframe())

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
            shuffle=False,
            seed=42,
        )
        train_dataset: Dataset = train_val['train']
        val_dataset: Dataset = train_val['test']
        datadict: DatasetDict = DatasetDict({'train': train_dataset, 'val': val_dataset})

        return datadict.map(
            function=lambda batch: WikipediaDataset.__batch_preprocess_training(tokenizer, batch), 
            batched=True, 
            batch_size=1024,
            num_proc=4,
        )
    
    def for_inference(self) -> DatasetDict:
        # get the raw dataset
        dataset: Dataset = self.get_raw_dataset(None)
        return DatasetDict({'all': dataset}).map(
            function=WikipediaDataset.__batch_preprocess_inference,
            batched=True,
            batch_size=1024,
            num_proc=4,
        )

    @staticmethod
    def __batch_preprocess_inference(batch: Dataset) -> Dataset:
        batch['text'] = [WikipediaDataset.__preprocess(doc) for doc in batch['text']]
        return batch

    @staticmethod
    def __batch_preprocess_training(tokenizer: T5Tokenizer, batch: Dataset) -> Dataset:
        batch['text'] = [
            'summarize: ' + text.replace(summary, '') if len(text) >= 3000 else text
            for text, summary in zip(batch['text'], batch['summary'])
        ]
        batch['text'] = [WikipediaDataset.__preprocess(doc) for doc in batch['text']]
        batch['summary'] = [WikipediaDataset.__preprocess(doc) for doc in batch['summary']]
        batch['input_ids'], batch['attention_mask'] = WikipediaDataset.__tokenize(tokenizer, batch['text'], max_length=512)
        batch['labels'], _ = WikipediaDataset.__tokenize(tokenizer, batch['summary'], max_length=512)
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
    
    @staticmethod
    def __preprocess(text):
        text: str = re.sub(r'\{\\displaystyle [^\}]+\}', '', text)
        text: str = re.sub(r'\{\\pdisplaystyle[^\}]+\}', '', text)
        text: str = re.sub(r'\{[^}]*\}', '', text)
        text: str = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text: str = re.sub(r'\\[a-zA-Z]+', '', text)
        text: str = re.sub(r'[\n\s]+', ' ', text)
        return text

    
if __name__ == '__main__':
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base', model_max_length=512)
    wiki = WikipediaDataset()
    # dataset = wiki.get_raw_dataset()
    training_dataset: DatasetDict = wiki.for_training(tokenizer, n_articles=1000)
    inference_dataset: DatasetDict = wiki.for_inference()


