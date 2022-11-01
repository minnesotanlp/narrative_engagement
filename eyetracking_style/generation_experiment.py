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
             options[1] + ": "
    # prompt = ' '.join(important_words)
    #if not important_words:
    #    prompt = text
    #result = classifier(prompt, options)
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
    return response.to_dict_recursive()


with open(STIMULI_FILENAME) as f, open('openai_results.csv', 'w') as out:
    reader = csv.reader(f)
    writer = csv.reader(out)
    stims = list(reader)[1:2]  # first line is header

    eye_measure_names = [name for name in eye_measures.keys()]
    metrics = eye_measure_names + ['cptm', 'ppl', 'hb']
    style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

    results = {'baseline': []}
    for metric in metrics:
        results[metric] = []

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

        # figure out if we have enough data for stim
        enough_data = True
        for i, measure in enumerate(metrics):
            if measure in eye_measure_names:
                data, con, incon = get_statistic(df, stim_id, text, eye_measures[measure])
            if len(data) != len(cleaned_text):
                enough_data = False
        for data in [hb, cptm, ppl]:
            if len(data) != len(cleaned_text):
                enough_data = False

        if not enough_data:
            continue

        # do baseline
        scores = get_response(text, [], options)
        result = scores[0] if style == options[0] else scores[1]
        results["baseline"].append(result)

        for i, measure in enumerate(metrics):
            hb = [abs(n) for n in tolist(stim[6])]
            cptm = tolist(stim[-6])
            incon_ppl = tolist(stim[-1])[4:]
            ppl = tolist(stim[-2])[4:]
            if measure in eye_measure_names:
                data, con, incon = get_statistic(df, stim_id, text, eye_measures[measure])
                std = np.nanstd(df[eye_measures[measure][1]])
                mean = np.nanmean(df[eye_measures[measure][1]])
            else:
                if measure == 'hb':
                    data = hb
                    std = all_hb.std()
                    mean = 0.4
                if measure == 'cptm':
                    data = cptm
                    std = all_hb.std()
                    mean = all_hb.mean()
                if measure == 'ppl':
                    data = ppl
                    std = all_ppl.std()
                    mean = all_ppl.mean()

            if len(data) != len(cleaned_text):
                continue
            indices = find_move_indices(cleaned_text)
            move_indices([data], indices)
            data = np.asarray(data)
            nonzero_data = np.where(data > 0, data, np.nan)
            data = np.where(data > mean + 0.5*std, data, 0)
            important_words = []
            for i in range(len(cleaned_text)):
                if data[i] > 0:
                    important_words.append(cleaned_text[i])
            scores = get_response(text, important_words, options)
            result = scores[0] if style == options[0] else scores[1]
            results[measure].append(result)

    for name, scores in results.items():
        print(name)
        print(sum(scores)/len(scores))
        correct = [score for score in scores if score >= 0.5]
        print(len(correct))
        print(sum(correct)/len(correct))
        incorrect = [score for score in scores if score < 0.5]
        print(len(incorrect))
        print(sum(incorrect)/len(incorrect))
        print()