import json
import csv
import math
from matplotlib import pyplot as plt

RESPONSE_FILE = 'openai_responses.csv'


def score_response(resp, label):
    label = label.lower()
    logprob_dicts = resp["choices"][0]["logprobs"]["top_logprobs"]
    best_res = 0
    for logprobs in logprob_dicts:
        for word, prob in logprobs.items():
            word = word.lower().strip()
            if label in word or label[0:3] == word:
                prob = math.exp(prob)
                if prob > best_res:
                    best_res = prob
    return best_res


def correct_response(resp, correct_label, incorrect_label):
    correct_prob = score_response(resp, correct_label)
    incorrect_prob = score_response(resp, incorrect_label)
    return correct_prob > incorrect_prob


with open(RESPONSE_FILE) as f:
    reader = csv.reader(f)
    lines = list(reader)
    header = lines[0]
    response_types = header[3:]

    score_results = {}
    opposite_score_results = {}
    accuracy_results = {}
    style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]
    for response_type in response_types:
        score_results[response_type] = []
        accuracy_results[response_type] = []
        opposite_score_results[response_type] = []

    for line in lines[2:]:
        stim_id = line[0]
        text = line[1]
        style = line[2]
        options = style_options[0] if style in style_options else style_options[1]
        incorrect = options[0] if style != options[0] else options[1]
        responses = line[3:]

        all_methods = True
        for response in responses:
            if len(response) == 0 or response == "NA":
                all_methods = False

        if not all_methods:
            continue

        for response, method in zip(responses, response_types):
            if len(response) == 0 or response == "NA":
                continue
            response_dict = json.loads(response)
            score_results[method].append(score_response(response_dict, style))
            opposite_score_results[method].append(score_response(response_dict, incorrect))
            accuracy_results[method].append(1 if correct_response(response_dict, style, incorrect) else 0)
    scores = []
    for method, num in score_results.items():
        print(method)
    for method, num in score_results.items():
        print(sum(num)/len(num))
        scores.append(sum(num)/len(num))
    print()
    for method, num in opposite_score_results.items():
        print(method, sum(num)/len(num))
    print()
    for method, num in accuracy_results.items():
        print(method, sum(num)/len(num))

    fig = plt.figure()
    ax = plt.axes()
    methods = [method[:-9] for method, _ in score_results.items()]
    print(scores)
    ax.bar([1, 2, 3, 4, 5, 6, 7, 8, 9], scores, width=0.5, tick_label=methods)
    plt.tight_layout()
    plt.show()