import os
from typing import List

from sentsplit.segment import SentSplit


def write_lines(lines, file_path):
    with open(file_path, 'w') as outf:
        for line in lines:
            outf.write(line + '\n')


def read_lines(file_path):
    lines = []
    with open(file_path, 'r') as inf:
        for line in inf:
            lines.append(line.rstrip('\n'))
    return lines


def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class SentenceSegmenter:
    def __init__(self) -> None:
        my_config = {
            'ngram': 5,
            'mincut': 7,
            'maxcut': 800,
            'strip_spaces': False,
            'segment_regexes': [
                {'name': 'newline'},
            ],
            'prevent_regexes': [
                {'name': 'liberal_url'},
                {'name': 'period_followed_by_lowercase'},
            ],
            'handle_multiple_spaces': True,
            'prevent_word_split': True,
        }
        self.sent_splitter = SentSplit('en', **my_config)

    def segment_from_text_file(self, text_file_path: str) -> List[str]:
        assert os.path.isfile(text_file_path)
        lines = [line.strip() for line in read_lines(text_file_path) if len(line.strip()) > 0]
        sents = self.sent_splitter.segment(' '.join(lines), strip_spaces=True)
        return sents
