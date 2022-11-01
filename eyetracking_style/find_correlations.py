from scipy.stats import spearmanr, pearsonr, kendalltau, ttest_ind
from transformers import BertTokenizerFast
import torch
import numpy as np
import seaborn as sn
import pandas as pd
import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# DATA_FILES = {'dwell time subtracted': open('Dwell Time_subresults.txt'),
#               'dwell time congruent': open('Dwell Time_conresults.txt'),
#               'dwell time incongruent': open('Dwell Time_incresults.txt'),
#               'ffd subtracted': open('FFD_subresults.txt'),
#               'ffd congruent': open('FFD_conresults.txt'),
#               'ffd incongruent': open('FFD_incresults.txt')}
STIMULI_FILENAME = 'ppl_experiment.csv'
DATA_FILENAME = 'processed_rdata.csv'


def tolist(item):
    item = item.strip()
    if item == 'NA':
        return []
    result = [i if i != 'nan' else 0 for i in item.split(' ')]
    return [float(i) for i in result if i]


def is_constant(measure):
    prev = measure[0]
    for m in measure:
        if m != prev:
            return False
    return True


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
    words = text.split(' ')
    words = [w.lower().strip(' ,"\'.`\n?!@#¬†;():*.') for w in words]
    sig_words = [0 for _ in range(max_IA)]

    for ia_id in range(1, max_IA + 1):
        idx = ia_id - 1
        ia_df = stim_df.loc[stim_df['IA_ID'] == ia_id]
        cons = ia_df.loc[ia_df['congruent'] == True]
        con[idx] = cons[measure_name].mean(skipna=True)
        incons = ia_df.loc[ia_df['congruent'] == False]
        incon[idx] = incons[measure_name].mean(skipna=True)
        sub[idx] = incons[sub_name].mean(skipna=True) - cons[sub_name].mean(skipna=True)
        p = ttest_ind(con, incon, alternative='less')[1]
        if p < 0.05:
            print(measure)
            sig_words[ia_id - 1] = 1


    idx_to_move = []
    if len(words) == max_IA:
        for ia_id in range(max_IA):
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
    else:
        print('oh no!!!!!!!!!!!!!')
        for ia in range(1, max_IA + 1):

            ia_df = stim_df.loc[stim_df['IA_ID'] == ia]

            print(ia_df['IA_LABEL'].tolist())

        print(words)
        print(len(words))
        print(max_IA)
    print()
    return sub, con, incon, sig_words, idx_to_move


def flatten(l):
    return [item for sublist in l for item in sublist]


def all_same_length(list_of_lists):
    assert(len(list_of_lists) > 0)
    last_length = len(list_of_lists[0])
    for sublist in list_of_lists:
        if len(sublist) != last_length:
            return False
    return True


def sim(first, second):
    same = 0
    for a, b in zip(first, second):
        if a == b:
            same += 1
        if type(a) == float:
            print(a)
    return same/len(first)



def move_indices(datas, idx_to_move):
    for data in datas:
        for frm, to in idx_to_move:
            data[to] += data[frm]
        indices_to_rm = [frm for frm, to in idx_to_move]
        for idx in sorted(indices_to_rm, reverse=True):
            del data[idx]

with open(STIMULI_FILENAME) as f, open('labels.csv', 'w') as out:
    reader = csv.reader(f)
    stims = list(reader)[1:]
    df = pd.read_csv(DATA_FILENAME)
    writer = csv.writer(out)

    word = df['word'].tolist()
    congruent = df['congruent'].tolist()
    stimID = df['stim_id'].tolist()
    iaID = df['IA_ID'].tolist()

    eye_measures = {'dwell time': ['IA_DWELL_TIME', 'subDT'],
                    'first fixation duration': ['IA_FIRST_FIXATION_DURATION', 'subFFD'],
                    'first run dwell time': ['IA_FIRST_RUN_DWELL_TIME', 'subFRD'],
                    'last run dwell time': ['IA_LAST_RUN_DWELL_TIME', 'subLRD'],
                    'regressions out': ['IA_REGRESSION_OUT_FULL_COUNT', 'subRO']}

    metrics = {}
    for name in eye_measures:
        metrics[name] = []
        metrics[name + ' con'] = []
        metrics[name + ' incon'] = []
    metrics['hummingbird'] = []
    metrics['captum'] = []
    metrics['surprisal'] = []
    metrics['significant et words'] = []

    for stim in stims:
        stim_id = int(stim[0])
        style = stim[1]
        source = stim[2]
        text = stim[3].strip()
        hb = [abs(i) for i in tolist(stim[6])]
        cptm = tolist(stim[-6])
        incon_ppl = tolist(stim[-1])[4:]
        ppl = tolist(stim[-2])[4:]
        label = stim[16]
        conf = stim[-3]
        writer.writerow([])
        writer.writerow(text.split(' ') + [label, max(1 - float(conf), float(conf))])
        writer.writerow(hb)
        writer.writerow(cptm)
        writer.writerow(ppl)

        # for each ia, get each statistic
        have_written = False
        all_measures = []
        for measure in eye_measures:
            sub, con, incon, sig_words, idx_to_move = get_statistic(df, stim_id, text, eye_measures[measure])

            hb = [abs(n) for n in tolist(stim[6])]
            cptm = tolist(stim[-6])
            ppl = tolist(stim[-2])[4:]

            datas = [hb, cptm, ppl, sig_words]

            if all_same_length(datas):
                move_indices(datas, idx_to_move)
                metrics[measure] += sig_words
                metrics[measure + ' con'] += con
                metrics[measure + ' incon'] += incon
                have_written = True
                if len(all_measures) == 0:
                    all_measures = np.asarray(sig_words)
                else:
                    all_measures = np.logical_or(all_measures, sig_words)

        if have_written:
            hb = [abs(n) for n in tolist(stim[6])]
            cptm = tolist(stim[-6])
            ppl = tolist(stim[-2])[4:]
            move_indices([hb, cptm, ppl], idx_to_move)
            metrics['hummingbird'] += hb
            metrics['captum'] += cptm
            metrics['surprisal'] += ppl
            metrics['significant et words'] += list(all_measures)

    correlations = [[0 for m in metrics.keys()] for m in metrics.keys()]
    for i, metric1 in enumerate(metrics.keys()):
        for j, metric2 in enumerate(metrics.keys()):
            if 'con' in metric1 or 'con' in metric2:
                continue
            print(metric1, metric2)
            m1 = np.asarray([m if not np.isnan(m) else 0 for m in metrics[metric1]])
            m2 = np.asarray([m if not np.isnan(m) else 0 for m in metrics[metric2]])
            #m1 = np.where(m1 > 1*m1.std() + m1.mean(), m1/m1.std(), 0)
            #m2 = np.where(m2 > 1*m2.std() + m2.mean(), m2/m2.std(), 0)
            #result = pearsonr(m1, m2)[0]
            jaccm1 = np.where(m1 > np.nanmean(m1), 1, 0)
            print(jaccm1)
            jaccm2 = np.where(m2 > np.nanmean(m2), 1, 0)
            result = sim(jaccm1, jaccm2)
            correlations[i][j] = result

    # plot heatmap
    plt.figure(figsize=(10, 7))
    correlations = np.asarray(correlations)
    mask = np.ones_like(correlations)
    mask[np.tril_indices_from(mask)] = False
    ax = sn.heatmap(data=correlations, cmap="Blues", annot=True, mask=mask, xticklabels=metrics.keys(), yticklabels=metrics.keys())
    plt.tight_layout()
    plt.show()
