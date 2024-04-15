from typing import List, Dict
from functools import cached_property

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn


from datasets import Dataset
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5Model, pipeline

from dataset import WikipediaDataset


class Query2Articles:

    def __init__(self, dataset: Dataset) -> None:
        self.dataset: Dataset = dataset
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.tfidf_matrix: np.ndarray = self.vectorizer.fit_transform(raw_documents=self.dataset['topic']).toarray() # shape: (N, V)
    
    def similarity(self, query: str) -> np.ndarray:
        query_vector: np.ndarray = self.vectorizer.transform([query]).toarray()  # shape: (1, V)
        return cosine_similarity(X=query_vector, Y=self.tfidf_matrix, dense_output=True) # shape (1, N)
    
    def rank_articles(self, query: str, n_articles: int = 1) -> List[str]:
        similarity_scores: np.ndarray = self.similarity(query=query).squeeze(axis=0)
        top_indices: np.ndarray = similarity_scores.argsort()[::-1]
        top_articles: np.ndarray = np.array(self.dataset['text'])[top_indices[:n_articles]]
        return list(top_articles)


if __name__ == '__main__':
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('.checkpoint/20240414142103/checkpoint-45627')
    wiki: WikipediaDataset = WikipediaDataset(csv_path='datasets/WikiData.csv', tokenizer=tokenizer)
    dataset: Dataset = wiki(split_ratios=None)
    mapper: Query2Articles = Query2Articles(dataset['all'])
    articles = mapper.rank_articles('What is DevOps', n_articles=1)

    checkpoint: str = '.checkpoint/20240414142103/checkpoint-91254'
    model = pipeline("summarization", model=checkpoint)
    output_text: str = model('\n'.join(articles), max_length=1000, min_length=100, do_sample=False)

    

