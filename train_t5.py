import os
import argparse
import datetime as dt
from typing import Dict, Any

import yaml

from transformers import (
    T5Tokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
)

from src.t5 import T5Model
from src.dataset import WikipediaDataset
from src.metrics import Rouge


def main(config: Dict[str, Any]) -> None:

    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        'google-t5/' + config['model']['tokenizer'], 
        model_max_length=5120,
    )

    if config['dataset']['name'] == 'Wikipedia':
        wiki = WikipediaDataset(
            csv_path=config['dataset']['csv_path'], 
            tokenizer=tokenizer, 
            n_articles=config['dataset']['n_articles'],
        )
        dataset = wiki(split_ratios=tuple(config['dataset']['split']))
    else:
        raise ValueError(f"Dataset {config['dataset']['name']} not available")

    if config['model']['name'].lower() == 't5':
        model: T5Model = T5Model(
            d_model=config['model']['d_model'],
            d_ff=config['model']['d_ff'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            relative_attention_num_buckets=config['model']['relative_attention_num_buckets'],
            dropout_rate=config['model']['dropout_rate'],
            initializer_factor=config['model']['initializer_factor'],
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Model {config['model']['name']} not supported")
    
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

    parser.add_argument('--config', type=str, default='t5.yaml', help='Configuration file name.')
    parser.add_argument('--dataset', type=str, choices=['Wikipedia',], help='Dataset\'s name')
    parser.add_argument('--csv_path', type=str, help='Path to the dataset file')
    parser.add_argument('--n_articles', type=int, help='How many articles are used for both training and evaluation')
    parser.add_argument('--split', nargs=3, type=float, help='List of split ratios')
    parser.add_argument('--model', type=str, help='Model\'s name')
    parser.add_argument('--tokenizer', type=str, help='Tokenizer\'s name')
    parser.add_argument('--d_model', type=int, help='Embedding dimension in nn.Embedding')
    parser.add_argument('--d_ff', type=int, help='Output dimenshion in nn.Linear')
    parser.add_argument('--num_layers', type=int, help='Number of layers in encoder and decoder, each layer has self-attention and feed-forward')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--relative_attention_num_buckets', type=int, help='Number of buckets used for relative positions in the self-attention mechanism')
    parser.add_argument('--dropout_rate', type=float, help='Dropout rate of the transformer')
    parser.add_argument('--initializer_factor', type=float, help='Scaling factor of the standard deviation in the normal initializer')
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
        if arg in ['dataset', 'model']:
            config[arg]['name'] = value
        elif arg in ['csv_path', 'split', 'n_articles']:
            config['dataset'][arg] = value
        elif arg in [
            'tokenizer', 'd_model', 'd_ff', 'num_layers', 
            'num_heads', 'relative_attention_num_buckets', 
            'dropout_rate', 'initializer_factor',
        ]:
            config['model'][arg] = value
        elif arg in [
            'output_dir', 'evaluation_strategy', 'learning_rate',
            'per_device_train_batch_size', 'per_device_eval_batch_size',
            'num_train_epochs', 'fp16',
        ]:
            config['training'][arg] = value
            
    # Run the main function with the configuration
    main(config)



