import os

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

user = os.environ["USER"]

def gpt2decode(tokens):
    return tokenizer.decode(tokens)

def gpt2encode(text):
    return tokenizer.encode(text)
