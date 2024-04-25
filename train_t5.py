import os
import argparse
import datetime as dt
from typing import Dict, Any

import yaml
import torch

from transformers import (
    T5Tokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    AutoModelForSeq2SeqLM,
)

from src.t5 import T5Base
from src.dataset import WikipediaDataset
from src.metrics import Rouge


def main(config: Dict[str, Any]) -> None:

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    if config['training']['from_checkpoint'] is not None:
        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(config['training']['from_checkpoint'])
        model = AutoModelForSeq2SeqLM.from_pretrained(config['training']['from_checkpoint'])
    else:
        tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
        model: T5Base = T5Base(tokenizer=tokenizer)
        model.load_state_dict(torch.load('t5_state_dict.pth'))

    wiki = WikipediaDataset(csv_path=config['dataset']['csv_path'])
    dataset = wiki.for_training(
        tokenizer=tokenizer, 
        split_ratios=tuple(config['dataset']['split']), 
        n_articles=config['dataset']['n_articles']
    )
    
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        output_dir=config['training']['output_dir'] + f'/{dt.datetime.now().strftime("%Y%m%d%H%M%S")}',
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['evaluation_strategy'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        num_train_epochs=config['training']['num_train_epochs'],
        fp16=config['training']['fp16'],
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='rougeLsum',
        greater_is_better=True,
    )

    rouge: Rouge = Rouge(tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: rouge(prediction_ids=x[0], label_ids=x[1]),   # x: transformers.EvalPrediction
    )
    trainer.train()


if __name__ == '__main__':

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train a T5 model for text summarization')

    parser.add_argument('--from_checkpoint', type=str, help='Continue training from a checkpoint.')
    parser.add_argument('--config', type=str, default='t5.yaml', help='Configuration file name.')
    parser.add_argument('--csv_path', type=str, help='Path to the dataset file')
    parser.add_argument('--n_articles', type=int, help='How many articles are used for both training and evaluation')
    parser.add_argument('--split', nargs=3, type=float, help='List of split ratios')
    parser.add_argument('--output_dir', type=str, help='Directory to save the model')
    parser.add_argument('--evaluation_strategy', type=str, choices=['no', 'steps', 'epoch'], help='Evaluate the model after each batch (`steps`) or after each epoch (`epoch`) or not evaluate the model at all (`no`)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--per_device_train_batch_size', type=int, help='Batch size per CPU/GPU for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Batch size per CPU/GPU for evaluating')
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--fp16', type=bool, help='Whether to use half-precision weight values')

    args: argparse.Namespace = parser.parse_args()

    # Load the configuration
    with open(file=f'{os.environ["PYTHONPATH"]}/configs/{args.config}', mode='r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Override the configuration with command-line arguments
    arg: str
    for arg in vars(args).keys():
        value: str | int | float | None = getattr(args, arg)
        if value is None:
            continue
        elif arg in ['csv_path', 'split', 'n_articles']:
            config['dataset'][arg] = value
        elif arg in [
            'from_checkpoint',
            'output_dir', 'evaluation_strategy', 'learning_rate',
            'per_device_train_batch_size', 'per_device_eval_batch_size',
            'num_train_epochs', 'fp16',
        ]:
            config['training'][arg] = value
            
    # Run the main function with the configuration
    main(config)



