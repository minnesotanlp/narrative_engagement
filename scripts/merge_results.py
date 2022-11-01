import json
import os
from collections import defaultdict

from utils import read_lines

if __name__ == '__main__':
    # Output directory to store resulting files in
    output_directory = '../data/kelsey/results'

    book_ids = ['expensivelessons', 'chemistswife', 'schoolmistress']
    for bn in book_ids:
        input_file = os.path.join(output_directory, f'{bn}_new_sentences.txt')
        sents = read_lines(input_file)

        token_file = os.path.join(output_directory, f'{bn}.tokens')
        lines = read_lines(token_file)
        token_idx_to_sent_idx = {}
        sent_idx_to_event_token_idx = defaultdict(list)
        for i, line in enumerate(lines):
            if i < 1:
                continue
            cols = line.split('\t')
            sent_idx = int(cols[1])
            token_idx = int(cols[3])
            token_idx_to_sent_idx[token_idx] = sent_idx

            if cols[-1] == 'EVENT':
                sent_idx_to_event_token_idx[sent_idx].append(token_idx)

        entity_file = os.path.join(output_directory, f'{bn}.entities')
        lines = read_lines(entity_file)
        sent_idx_to_corefs = defaultdict(list)
        for i, line in enumerate(lines):
            if i < 1:
                continue
            cols = line.split('\t')
            # We only look at PERSON
            if cols[4] != 'PER':
                continue
            coref = int(cols[0])
            token_idx = int(cols[2])
            text = cols[-1]
            sent_idx = token_idx_to_sent_idx[token_idx]
            sent_idx_to_corefs[sent_idx].append({'entity_id': coref, 'text':text.strip()})

        sentiment_file = os.path.join(output_directory, f'{bn}.sentiment')
        sentiments = read_lines(sentiment_file)

        emotion_file = os.path.join(output_directory, f'{bn}.emotion')
        emotions = read_lines(emotion_file)

        supersense_file = os.path.join(output_directory, f'{bn}.supersense')
        supersenses = read_lines(supersense_file)
        sent_idx_to_supersense = defaultdict(list)
        for i, line in enumerate(supersenses):
            if i < 1:
                continue
            start_token, end_token, supersense_category, text = line.split('\t')
            start_token = int(start_token)
            end_token = int(end_token)
            sent_idx = token_idx_to_sent_idx[start_token]
            sent_idx_to_supersense[sent_idx].append({'start_token_id': start_token, 'end_token_id': end_token, 'supersense_category': supersense_category, 'text': text})

        results = []
        for i, sent in enumerate(sents):
            negative, neutral, positive = sentiments[i].split('\t')
            sent_result = {
                'sentence': sent,
                'negative_sentiment': float(negative),
                'neutral_sentiment': float(neutral),
                'positive_sentiment': float(positive),
                'persons': sent_idx_to_corefs[i],
                'event_token_ids': sent_idx_to_event_token_idx[i],
                'supersense': sent_idx_to_supersense[i],
            }
            from pprint import pprint
            pprint(sent_result)
            results.append(sent_result)

        with open(os.path.join(output_directory, f'{bn}_nlp_features.jsonl'), 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write('\n')
