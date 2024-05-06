import os
from typing import Tuple, List, Dict, Any

import yaml
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from datasets import DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

from .dataset import WikipediaDataset


class Agent:

    def __init__(self, config: str, checkpoint: str) -> None:
        # Config
        with open(file=config, mode='r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        # Data
        self.wiki: WikipediaDataset = WikipediaDataset()
        self.dataset: DatasetDict = self.wiki.for_inference()
        # Tokenizer & Model
        self.checkpoint: str = checkpoint
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(checkpoint)
        # Mapper
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.tfidf_matrix: NDArray = self.vectorizer.fit_transform(raw_documents=self.dataset['all']['topic']).toarray() # shape: (N, V)
    
    def similarity(self, query: str) -> NDArray:
        query_vector: NDArray = self.vectorizer.transform([query]).toarray()  # shape: (1, V)
        return cosine_similarity(X=query_vector, Y=self.tfidf_matrix, dense_output=True) # shape (1, N)
    
    def rank_articles(self, query: str, n_articles: int = 1) -> Tuple[List[str], List[str]]:
        similarity_scores: NDArray = self.similarity(query=query).squeeze(axis=0)
        top_indices: NDArray = similarity_scores.argsort()[::-1]
        top_articles: NDArray = np.array(self.dataset['all']['text'])[top_indices[:n_articles]]
        top_urls: NDArray = np.array(self.dataset['all']['url'])[top_indices[:n_articles]]
        return list(top_urls), list(top_articles)
    
    def run(self):

        while True:
            query: str = input('-------\nPlease ask me anything about Computer Science:\n > Question: ')
            if query == ':q':
                print('Thank you for using our service')
                break
            if len(query) < 10:
                print('Please clarify your question')
                continue
            
            query: str = query.lower()
            for prefix in [
                'explain to me', 'explain', 
                'tell me about', 'say something about',
                'talk about', 'help me understand',
                'what is', 'what are', 'what',
            ]:
                query: str = query.lstrip(prefix)
            urls, articles = self.rank_articles(query, n_articles=1)
            inputs: torch.Tensor = self.tokenizer(
                '\n'.join(articles), 
                return_tensors="pt", 
                max_length=self.tokenizer.model_max_length, 
                truncation=True
            ).input_ids
            outputs: torch.Tensor = self.model.generate(
                inputs, 
                max_new_tokens=self.config['agent']['max_summary_length'], 
                do_sample=False
            )
            response: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f'Retrieving information from {" ".join(urls)} ...')
            print(response)




