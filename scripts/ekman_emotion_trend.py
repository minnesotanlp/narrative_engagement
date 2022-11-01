import json
import os
from typing import List

import more_itertools
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import SentenceSegmenter, chunk_list, read_lines


def apply_fft_low_pass_filter_and_inverse(signal):
    fft_signal = fft(np.array(signal))
    # need to study more on this Butterworth filtering..
    order = 6
    fs = 30.0
    cutoff = 3.667
    b, a = butter(order, cutoff, fs=fs)
    zi = lfilter_zi(b, a)
    filtered_fft_signal, _ = lfilter(b, a, fft_signal, zi=zi)
    inversed_signal = ifft(filtered_fft_signal)
    # discard leftover imaginary parts
    inversed_signal = [n.real for n in inversed_signal]
    return inversed_signal

class EmotionTrend:
    def __init__(self) -> None:
        model_name = 'j-hartmann/emotion-english-distilroberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    def classify(self, sentences: List[str], num_sents_per_batch=50, top_k=3):
        assert isinstance(sentences, list)
        all_scores = []
        for sents in chunk_list(sentences, num_sents_per_batch):
            encoded_input = self.tokenizer(sents, return_tensors='pt', truncation=True, padding=True, max_length=250)
            output = self.model(**encoded_input)
            scores = output[0].detach().numpy()
            scores = softmax(scores, axis=1)
            for scores_per_sent in scores:
                scores_per_sent = {label: score.item() for label, score in zip(self.emotions, scores_per_sent)}
                # scores_per_sent = sorted(scores_per_sent, reverse=True, key=lambda x: x[1])[:top_k]
                all_scores.append(scores_per_sent)
            # print(len(all_scores))
        return all_scores

    def classify_windowed_sentences(self, sentences: List[str], window: int = 10, do_fft: bool = False):

        scores = self.classify(sentences)
        if do_fft:
            neg_scores = []
            neu_scores = []
            pos_scores = []
            for neg, neu, pos in scores:
                neg_scores.append(neg)
                neu_scores.append(neu)
                pos_scores.append(pos)
            neg_scores = _apply_fft_low_pass_filter_and_inverse(neg_scores)
            neu_scores = _apply_fft_low_pass_filter_and_inverse(neu_scores)
            pos_scores = _apply_fft_low_pass_filter_and_inverse(pos_scores)
            scores = [[neg, neu, pos] for neg, neu, pos in zip(neg_scores, neu_scores, pos_scores)]

        windowed_scores = list(more_itertools.windowed(scores, window))
        windowed_avg = []
        for w in windowed_scores:
            window = np.asarray(w)
            avg = np.mean(window, axis=0).tolist()
            windowed_avg.append(avg)
        return windowed_avg


if __name__ == '__main__':
    emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    output_directory = '../data/kelsey/results'
    book_ids = ['expensivelessons', 'chemistswife', 'schoolmistress']
    # book_ids = ['expensivelessons']
    # for bn in book_ids:
    #     # {bn}_new_sentences are present after running the `run_booknlp.py`
    #     input_file = os.path.join(output_directory, f'{bn}_new_sentences.txt')
    #     sents = read_lines(input_file)
    #     emotion_trend = EmotionTrend()
    #     emotion_scores = emotion_trend.classify(sents)
    #     with open(os.path.join(output_directory, f'{bn}.emotion'), 'w') as f:
    #         for emotions in emotion_scores:
    #             f.write(f'{json.dumps(emotions)}\n')

    window = 5
    for bn in book_ids:
        emotion_scores = {k: [] for k in emotion_classes}
        path = os.path.join(output_directory, f'{bn}.emotion')
        emotions = [json.loads(line) for line in read_lines(path)]
        windowed_emotions = list(more_itertools.windowed(emotions, window))
        windowed_avg = []
        for w in windowed_emotions:
            # w: [{}, {}, {}]
            for emo_cls in emotion_classes:
                scores = [e[emo_cls] for e in w]
                emotion_scores[emo_cls].append(np.mean(scores))

        df = pd.DataFrame(emotion_scores, columns=emotion_classes)
        # print(df)
        for emo_cls in emotion_classes:
            if emo_cls == 'neutral':
                continue
            line_plot = sns.lineplot(data=df[['neutral', emo_cls]])
            fig = line_plot.get_figure()
            fig.savefig(f'ekman_line_plot_{bn}_{window}_{emo_cls}.png')
            plt.clf()

    # Entity grounding: https://spacy.io/usage/linguistic-features#entity-linking but needs to train our own model
    # Event extraction: no readily usable model exists, could try: https://github.com/luyaojie/Text2Event
