from transformers import T5Config
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Model(T5ForConditionalGeneration):

    def __init__(
        self, 
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        relative_attention_num_buckets: int,
        dropout_rate: float,
        initializer_factor: float,
        tokenizer: T5Tokenizer,
    ):
        self.tokenizer: T5Tokenizer = tokenizer
        self.config: T5Config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            dropout_rate=dropout_rate,
            initializer_factor=initializer_factor,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        super().__init__(self.config)
        
        
