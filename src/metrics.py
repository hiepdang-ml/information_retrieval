from typing import TYPE_CHECKING, Dict, List

import torch

import evaluate
from evaluate import EvaluationModule
from transformers import T5Tokenizer


class Rouge:

    def __init__(self, tokenizer: T5Tokenizer) -> None:
        self.rouge: EvaluationModule = evaluate.load(path="rouge")
        self.tokenizer: T5Tokenizer = tokenizer

    def __call__(self, prediction_ids: torch.Tensor, label_ids: torch.Tensor) -> Dict:
        prediction_ids: torch.Tensor = torch.where(
            condition=prediction_ids != -100,
            input=prediction_ids,
            other=self.tokenizer.pad_token_id
        )
        decoded_preds: List[str] = self.tokenizer.batch_decode(
            sequences=prediction_ids,
            skip_special_tokens=True
        )
        label_ids: torch.Tensor = torch.where(
            condition=label_ids != -100,
            input=label_ids,
            other=self.tokenizer.pad_token_id
        )
        decoded_labels: List[str] = self.tokenizer.batch_decode(
            sequences=label_ids,
            skip_special_tokens=True
        )
        result: Dict[str, torch.Tensor] = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        generation_lengths = [
            torch.count_nonzero(prediction_id != self.tokenizer.pad_token_id)
            for prediction_id in prediction_ids
        ]
        result["generation_lengths"] = torch.tensor(generation_lengths).half().mean()

        return {k: round(v.item(), 4) for k, v in result.items()}


if __name__ == '__main__':

    preds: List[str] = ['i love you', 'you hate me']
    labels: List[str] = ['i miss you', 'you do not like me']
    rouge: Rouge = Rouge(tokenizer=T5Tokenizer.from_pretrained('google-t5/t5-large'))
    preds_encoding = rouge.tokenizer(preds, return_tensors='pt', padding=True, truncation=True)
    labels_encoding = rouge.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)

    d = rouge(
        prediction_ids=preds_encoding.input_ids, 
        label_ids=labels_encoding.input_ids
    )



