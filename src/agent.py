import os
from typing import List, Dict, Any
from functools import cached_property

import yaml
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn


from datasets import DatasetDict
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5Model, pipeline

from .dataset import WikipediaDataset


class Agent:

    def __init__(self, config: str, csv_path: str, checkpoint: str) -> None:
        # Config
        with open(file=config, mode='r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        # Data
        self.csv_path: str = csv_path
        self.wiki: WikipediaDataset = WikipediaDataset(csv_path=csv_path)
        self.dataset: DatasetDict = self.wiki.for_inference(max_article_length=self.config['agent']['max_article_length'])
        # Model
        self.checkpoint: str = checkpoint
        self.model = pipeline(task='summarization', model=checkpoint)
        # Mapper
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.tfidf_matrix: NDArray = self.vectorizer.fit_transform(raw_documents=self.dataset['all']['topic']).toarray() # shape: (N, V)
    
    def similarity(self, query: str) -> NDArray:
        query_vector: NDArray = self.vectorizer.transform([query]).toarray()  # shape: (1, V)
        return cosine_similarity(X=query_vector, Y=self.tfidf_matrix, dense_output=True) # shape (1, N)
    
    def rank_articles(self, query: str, n_articles: int = 1) -> List[str]:
        similarity_scores: NDArray = self.similarity(query=query).squeeze(axis=0)
        top_indices: NDArray = similarity_scores.argsort()[::-1]
        top_articles: NDArray = np.array(self.dataset['all']['text'])[top_indices[:n_articles]]
        return list(top_articles)
    
    def run(self):

        while True:
            query: str = input('-------\nPlease ask me anything about Computer Science:\n > Question: ')
            if query == ':q':
                print('Thank you for using our service')
                break
            if len(query) < 10:
                print('Please clarify your question')
                continue
            
            articles: List[str] = self.rank_articles(query, n_articles=1)
            response: str = self.model(
                '\n'.join(articles), 
                max_length=self.config['agent']['max_summary_length'], 
                min_length=self.config['agent']['max_summary_length'], 
                do_sample=False,
            )[0]['summary_text']
            print(response)




