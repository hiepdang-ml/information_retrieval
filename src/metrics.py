from typing import Dict, List

import numpy as np
import torch

import evaluate
from evaluate import EvaluationModule
from transformers import T5Tokenizer


class Rouge:

    def __init__(self, tokenizer: T5Tokenizer) -> None:
        self.rouge: EvaluationModule = evaluate.load(path="rouge")
        self.tokenizer: T5Tokenizer = tokenizer

    def __call__(self, prediction_ids: np.ndarray, label_ids: np.ndarray) -> Dict:
        prediction_ids: np.ndarray = np.where(prediction_ids != -100, prediction_ids, self.tokenizer.pad_token_id)
        decoded_preds: List[str] = self.tokenizer.batch_decode(
            sequences=prediction_ids,
            skip_special_tokens=True
        )
        label_ids: np.ndarray = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_labels: List[str] = self.tokenizer.batch_decode(
            sequences=label_ids,
            skip_special_tokens=True
        )
        result: Dict[str, np.ndarray] = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        generation_lengths = [
            np.count_nonzero(prediction_id != self.tokenizer.pad_token_id)
            for prediction_id in prediction_ids
        ]
        result["generation_lengths"] = np.array(generation_lengths).mean()
        print(result)

        return {k: round(v, 4) for k, v in result.items()}




