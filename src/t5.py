from transformers import T5Config
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Base(T5ForConditionalGeneration):

    def __init__(self, tokenizer: T5Tokenizer):
        self.tokenizer: T5Tokenizer = tokenizer
        self.config: T5Config = T5Config(
            d_model=768,
            d_ff=3072,
            num_layers=12,
            num_heads=12,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            initializer_factor=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        super().__init__(self.config)


