from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizerFast
import torch

def get_ppl(encodings):
    scores = []
    for i in range(1, encodings.input_ids.size(1) + 1):
        input_ids = encodings.input_ids[:, 0:i]
        target_ids = input_ids.clone()
        target_ids[:, :i - 1] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs[0]
            scores.append(nll.item())
    return scores