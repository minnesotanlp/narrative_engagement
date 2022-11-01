import os
from typing import List

import more_itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import SentenceSegmenter, chunk_list, read_lines


class SentimentTrend:
    def __init__(self) -> None:
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify(self, sentences: List[str], num_sents_per_batch=50):
        assert isinstance(sentences, list)
        all_scores = []
        for sents in chunk_list(sentences, num_sents_per_batch):
            encoded_input = self.tokenizer(sents, return_tensors='pt', truncation=True, padding=True, max_length=250)
            output = self.model(**encoded_input)
            scores = output[0].detach().numpy()
            scores = softmax(scores, axis=1)
            all_scores.extend(scores)
        return all_scores

    def classify_windowed_sentences(self, sentences: List[str], window: int = 10, do_fft: bool = False):
        def _apply_fft_low_pass_filter_and_inverse(signal):
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
    output_directory = '../data/kelsey/results'
    # book_ids = ['expensivelessons', 'chemistswife', 'schoolmistress']
    book_ids = ['schoolmistress']
    for bn in book_ids:
        # {bn}_new_sentences are present after running the `run_booknlp.py`
        input_file = os.path.join(output_directory, f'{bn}_new_sentences.txt')
        sents = read_lines(input_file)
        sentiment_trend = SentimentTrend()
        # [(negative_score, neutral_score, positive_score), ..]
        sentiment_scores = sentiment_trend.classify(sents)
        with open(os.path.join(output_directory, f'{bn}.sentiment'), 'w') as f:
            for neg, neu, pos in sentiment_scores:
                f.write(f'{neg}\t{neu}\t{pos}\n')

    # doc_path = '../data/short_story/expensive_lessons.txt'
    # sentence_segmenter = SentenceSegmenter()
    # sents = sentence_segmenter.segment_from_text_file(doc_path)
    # print(len(sents))
    # # write_lines(sents, doc_path[:-3] + 'sents')

    # sentiment_trend = SentimentTrend()
    # # [(negative_score, neutral_score, positive_score), ..]
    # windowed_avg_scores = sentiment_trend.classify_windowed_sentences(sents, window=1, do_fft=False)
    # df = pd.DataFrame(windowed_avg_scores, columns=['negative', 'neutral', 'positive'])
    # df['negative'] = df['negative'].apply(lambda x: x * -1)
    # df['positive'] = df['positive'].apply(lambda x: x + 1)
    # df['summed'] = df['negative'] + df['neutral'] + df['positive']
    # print(df)
    # # line_plot = sns.lineplot(data=df, x=df.index, y='negative')
    # line_plot = sns.lineplot(data=df)
    # fig = line_plot.get_figure()
    # fig.savefig('line_plot.png')

    # Entity grounding: https://spacy.io/usage/linguistic-features#entity-linking but needs to train our own model
    # Event extraction: no readily usable model exists, could try: https://github.com/luyaojie/Text2Event
