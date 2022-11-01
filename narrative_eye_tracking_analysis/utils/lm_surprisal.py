from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizerFast
import torch
import csv
import numpy as np
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


def find_start_token(tokens, start_tokens)
    """Finds token after which the target sentence begins"""
    idx = len(tokens)
    prev_token = Null
    while idx > 0:
        idx -= 1
        if tokens[idx] == start_tokens[0] and prev_token == start_tokens[1]:
            break
        prev_token = tokens[idx]
    return idx


def get_substring_nll(encodings, start_idx):
    """Gets avg negative log likelihood of all tokens in a substring"""
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    target_ids[:, :start_idx] = -100  # ignore tokens before the action begins
    with torch.no_grad:
        outputs = model(input_ids, labels=target_ids)
    nll = outputs[0]
    return nll.item()


def score(context, sentence):
    """Returns LM surprisal score (negative log likelihood loss) for given sentence, conditioned on context"""
    prompt = context + sentence
    sentence_encoding = tokenizer(sentence, return_tensors='pt')
    encoding = tokenizer(prompt, return_tensors='pt')
    start_idx = encoding.input_ids.size(1) - sentence_encoding.input_ids.size(1)
    return get_nll(encoding, start_idx)


