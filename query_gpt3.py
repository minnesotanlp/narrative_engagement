import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
from transformers import pipeline, RobertaTokenizer, AutoModelForSequenceClassification
import numpy as np
import openai
import json

API_KEY = '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'
openai.api_key = API_KEY
DATA_FILENAME = 'processed_rdata.csv'
STIMULI_FILENAME = 'ppl_experiment.csv'
df = pd.read_csv(DATA_FILENAME)

eye_measures = {'dwell time': ['IA_DWELL_TIME', 'subDT'],
                'first fixation duration': ['IA_FIRST_FIXATION_DURATION', 'subFFD'],
                'first run dwell time': ['IA_FIRST_RUN_DWELL_TIME', 'subFRD'],
                'last run dwell time': ['IA_LAST_RUN_DWELL_TIME', 'subLRD'],
                'regressions out': ['IA_REGRESSION_OUT_FULL_COUNT', 'subRO']}

# classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-roberta-base")


def tolist(item):
    item = item.strip()
    if item == 'NA':
        return []
    result = [i if i != 'nan' else 0 for i in item.split(' ')]
    return [float(i) for i in result if i]


def get_max_IA(df, stim_id):
    ias = df.loc[(df['stim_id'] == stim_id)]['IA_ID'].tolist()
    return sorted(ias)[-1]


def get_statistic(df, stim_id, text, measure):
    """gets list of ia-ordered averaged metric for subtractive, congruent, and incongruent case"""
    max_IA = get_max_IA(df, stim_id)
    sub = [0 for _ in range(max_IA)]
    con = [0 for _ in range(max_IA)]
    incon = [0 for _ in range(max_IA)]

    measure_name, sub_name = measure

    stim_df = df.loc[(df['stim_id'] == stim_id)]

    for ia_id in range(1, max_IA + 1):
        idx = ia_id - 1
        ia_df = stim_df.loc[stim_df['IA_ID'] == ia_id]
        cons = ia_df.loc[ia_df['congruent'] == True]
        con[idx] = cons[measure_name].mean(skipna=True)
        incons = ia_df.loc[ia_df['congruent'] == False]
        incon[idx] = incons[measure_name].mean(skipna=True)
        sub[idx] = incons[sub_name].mean(skipna=True) - cons[sub_name].mean(skipna=True)

    return sub, con, incon


def clean_word(word):
    return word.lower().strip(' ,"\'.`\n?!@#¬†;():*.')


def is_constant(measure):
    prev = measure[0]
    for m in measure:
        if m != prev:
            return False
    return True


def flatten(l):
    return [item for sublist in l for item in sublist]


def find_move_indices(words):
    idx_to_move = []
    for ia_id in range(len(words)):
        if words[ia_id] in stopwords.words('english'):
            j = ia_id
            left_idx, right_idx = None, None
            while j >= 0:
                j -= 1
                if words[j] not in stopwords.words('english'):
                    left_idx = j
                    break
            j = ia_id
            while j < len(words):
                if words[j] not in stopwords.words('english'):
                    right_idx = j
                j += 1
            if left_idx and right_idx:
                idx = left_idx if ia_id - left_idx < ia_id - right_idx else right_idx  # right wins in case of tie
            else:
                idx = left_idx if left_idx else right_idx
            idx_to_move.append((ia_id, idx))
    return idx_to_move


def move_indices(datas, idx_to_move):
    for data in datas:
        for frm, to in idx_to_move:
            data[to] += data[frm]
            data[frm] = 0


def get_response(text, important_words, options):
    prompt = "Decide whether the text style is " + options[0] + " or " + options[1] \
            + "\nText: " + text + "\nImportant words: " + ' '.join(important_words) + "\n" + options[0] + " or " +\
             options[1] + ":"
    print(prompt)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        logprobs=5,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    return json.dumps(response.to_dict_recursive())

def get_baseline_response(text, options):
    prompt = "Decide whether the text style is " + options[0] + " or " + options[1] \
             + "\nText: " + text + "\n" + options[0] + " or " + \
             options[1] + ":"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        logprobs=5,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    return json.dumps(response.to_dict_recursive())


with open(STIMULI_FILENAME) as f, open('.csv', 'w') as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    header_row = ['stim_id', 'text', 'style']
    stims = list(reader)[1:]  # first line is header

    eye_measure_names = [name for name in eye_measures.keys()]
    metrics = eye_measure_names
    for metric in metrics:
        header_row.append(metric + ' response')
    writer.writerow(header_row)
    style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

    all_hb = np.asarray([abs(n) for n in flatten([tolist(line[6]) for line in stims])])
    all_cptm = np.asarray(flatten([tolist(line[-6]) for line in stims]))
    all_ppl = np.asarray(flatten([tolist(line[-2][4:]) for line in stims]))

    for stim in stims:
        stim_id = int(stim[0])
        style = stim[1].split(' ')[0]
        options = style_options[0] if style in style_options[0] else style_options[1]
        hb = [abs(n) for n in tolist(stim[6])]
        cptm = tolist(stim[-6])
        ppl = tolist(stim[-2])[4:]

        text = stim[3]
        cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
        out_line = [stim_id, text, style]

        for i, measure in enumerate(metrics):
            hb = [abs(n) for n in tolist(stim[6])]
            cptm = tolist(stim[-6])
            ppl = tolist(stim[-2])[4:]
            if measure in eye_measure_names:
                sub, con, data = get_statistic(df, stim_id, text, eye_measures[measure])
                std = np.nanstd(df[eye_measures[measure][0]])
                mean = np.nanmean(df[eye_measures[measure][0]])
            else:
                if measure == 'hb':
                    data = hb
                    std = all_hb.std()
                    mean = 0
                if measure == 'cptm':
                    data = cptm
                    std = all_hb.std()
                    mean = all_hb.mean()
                if measure == 'ppl':
                    data = ppl
                    std = all_ppl.std()
                    mean = all_ppl.mean()

            if len(data) != len(cleaned_text):
                out_line.append('NA')
                continue
            indices = find_move_indices(cleaned_text)
            move_indices([data], indices)
            data = np.asarray(data)
            nonzero_data = np.where(data > 0, data, np.nan)
            data = np.where(data > mean, data, 0)
            important_words = []
            for i in range(len(cleaned_text)):
                if data[i] > 0:
                    important_words.append(cleaned_text[i])
            result = ''
            if len(important_words) > 0:
                result = get_response(text, important_words, options)
            out_line.append(result)
        writer.writerow(out_line)
