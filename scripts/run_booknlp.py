import os

from booknlp.booknlp import BookNLP

from utils import write_lines


if __name__ == '__main__':

    corpus_directory = '../data/kelsey'
    # Output directory to store resulting files in
    output_directory = '../data/kelsey/results'

    book_ids = ['expensivelessons', 'chemistswife', 'schoolmistress']
    for bn in book_ids:
        file_path = os.path.join(corpus_directory, f'{bn}_sentences.txt')
        # File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
        # model_params = {
        #     'pipeline':'entity,quote,supersense,event,coref',
        #     'model':'big'
        # }
        # booknlp = BookNLP('en', model_params)
        # booknlp.process(file_path, output_directory, bn)

        # Re-organize sentences to keep consistent .tokens file
        with open(file_path, 'r') as f:
            original_doc = f.read()
        tokens_path = os.path.join(output_directory, f'{bn}.tokens')
        with open(tokens_path) as f:
            prev_sentence_id = 0
            prev_char_id = 0
            sentences = []
            for i, line in enumerate(f):
                if i < 1:
                    continue
                line = line.strip()
                cols = line.split('\t')
                assert len(cols) == 13
                sentence_id = int(cols[1])
                char_id = int(cols[6])
                if sentence_id != prev_sentence_id:
                    sentence = original_doc[prev_char_id:char_id].replace('\n', ' ').strip()
                    sentences.append(sentence)
                    prev_sentence_id = sentence_id
                    prev_char_id = char_id

            sentence = original_doc[prev_char_id:].replace('\n', ' ').strip()
            sentences.append(sentence)

        write_lines(sentences, os.path.join(output_directory, f'{bn}_new_sentences.txt'))
