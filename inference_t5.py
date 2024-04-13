import argparse

from transformers import pipeline

def main(checkpoint: str, input_text: str, input_file: str) -> str:

    model = pipeline("summarization", model=checkpoint)

    if input_text is not None:
        output_text = model('summarize: ' + input_text, max_length=1000, min_length=100, do_sample=False)
    elif input_file is not None:
        with open(input_file, mode='r') as file:
            output_text = model('summarize: ' + file.read(), max_length=1000, min_length=100, do_sample=False)
    else:
        raise ValueError('Either input_text or input_file must be specified')

    return output_text


if __name__ == '__main__':

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Train a T5 model for text summarization')

    parser.add_argument('--checkpoint', type=str, help='Trained model checkpoint')
    parser.add_argument('--input_text', type=str, help='Full-content input text')
    parser.add_argument('--input_file', type=str, help='Full-content input file')

    args: argparse.Namespace = parser.parse_args()

    print(main(**vars(args)))





