import pandas as pd
from nltk.corpus import stopwords


EYELINK_OUTPUT = 'IA923.txt'
PROCESSED_OUTPUT_DESTINATION = 'appended_923_skipstops.txt'  # filename of output for eyelink data with HAL_FREQ, LENGTH, WORD
# attributes added

# load data containing word frequencies in the stimuli's vocabulary
df = pd.read_csv('/Users/karin/Downloads/word_data - Sheet1.csv')
df['Word'] = df['Word'].str.lower()

prev_inc = False
prev_trial = 0
with open(EYELINK_OUTPUT) as f:
    lines = f.readlines()
    with open(PROCESSED_OUTPUT_DESTINATION, 'w') as out:
        # rewrite header line
        print(lines[0].split('\t'))
        out.write(lines[0][:-1] + '\tword\tHAL_FREQ\tLENGTH\n')
        lines = lines[1:]

        extra = []
        for line in lines:
            items = line.split('\t')
            word = items[11].lower().strip(' ,"\'.`\n?!@#¬†;():*')
            trial_index = int(items[1])
            if trial_index != prev_trial:
                skip_trial = prev_inc  # if prev trial was incongruent, discount this trial
            prev_inc = items[19] != 'True'
            prev_trial = trial_index
            entry = df.loc[df['Word'] == word]
            if len(entry) == 0:  # word not found, use NA for freq
                freq = '.'
                length = str(len(word))
            else:
                length = entry['Length'].item()
                freq = entry['HAL LOG FREG'].item()
            if line[-1] == '\n':
                line = line[:-1]
            if skip_trial:
                continue

            line = line + '\t' + word + '\t' + str(freq) + '\t' + str(length) + '\n'
            out.write(line)
