from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizerFast
import torch
import csv
import numpy as np
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

STIMULI_FILE = 'EyetrackingStimuli copy.csv'


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


def space_indices(text):
    result = []
    for i, char in enumerate(text):
        if char == ' ':
            result.append(i)
    return result


def convert_scores(encodings, scores, text):
    m = score_map(encodings, text)
    if len(m) != len(scores):
        print(len(m))
        print(len(scores))
        print(text)
        return []
    result = [0 for _ in text.split(' ')]
    for i, score in enumerate(scores):
        idx = m[i]
        result[idx] += float(score)
    return result


def score_map(encodings, text):
    spaces = space_indices(text) + [len(text)]
    word_ids = encodings.word_ids()
    token_map = []
    current_idx = 0
    for w in word_ids:
        if w is None:
            continue
        # update idx if we have hit a space in the original string
        chrs = encodings.word_to_chars(w)
        token_map.append(current_idx)
        if current_idx < len(text.split(' ')) - 1 and chrs.end >= spaces[current_idx]:
            current_idx += 1
    return token_map


# regular ppl calculations
def make_ppl_file():
    with open(STIMULI_FILE) as f, open('ppl.csv', 'w') as out:
        reader = csv.reader(f)
        writer = csv.writer(out)
        lines = list(reader)[1:]  # first line is header
        for line in lines:
            text = line[3]
            encodings = tokenizer(text, return_tensors='pt')
            ppl_scores = convert_scores(encodings, get_ppl(encodings), text)
            ppl_scores = ' '.join([str(s) if s is not np.nan else '0' for s in ppl_scores])
            captum = [float(i) for i in line[13].split(' ')[1:-1]]
            captum_scores = ' '.join([str(s) for s in convert_scores(bert_tokenizer(text), captum, text)])
            writer.writerow(line + [ppl_scores] + [captum_scores])


def do_context_ppl(style, text):
    text = 'The following text is ' + style + ': ' + text
    encodings = tokenizer(text, return_tensors='pt')
    return convert_scores(encodings, get_ppl(encodings), text)


def run_ppl_experiment():
    with open(STIMULI_FILE) as f, open('ppl_experiment.csv', 'w') as out:
        reader = csv.reader(f)
        writer = csv.writer(out)
        lines = list(reader)
        writer.writerow(lines[0] + ['congruent ppl', 'incon ppl'])
        lines = lines[1:]  # first line is header
        for line in lines:
            text = line[3]
            style = line[1].split(' ')[0]
            if 'Negative' in style:
                constyle = 'Positive'
            elif 'Positive' in style:
                constyle = 'Negative'
            elif 'Polite' in style:
                constyle = 'Impolite'
            elif 'Impolite' in style:
                constyle = 'Polite'
            con_ppl_scores = do_context_ppl(style, text)
            incon_ppl_scores = do_context_ppl(constyle, text)
            writer.writerow(line + [' '.join([str(s) if s is not np.nan else '0' for s in con_ppl_scores]),
                            ' '.join([str(s) if s is not np.nan else '0' for s in incon_ppl_scores])])

run_ppl_experiment()