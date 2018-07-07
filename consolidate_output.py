import argparse
import csv
import glob
import json
import os
from collections import defaultdict, OrderedDict
from typing import List

import numpy as np
import spacy
from textacy.text_stats import TextStats

spacy_parser = spacy.load('en_core_web_sm', disable=['ner', 'tagger'])


def get_scores(texts: List[str]):
    scores = [TextStats(doc).readability_stats for doc in spacy_parser.pipe(texts, batch_size=1024)]
    scores = list(filter(None, scores))

    return {
        key: np.mean(list(s[key] for s in scores))
        for key in scores[0].keys()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data-dir', required=True)
    parser.add_argument('-d', '--decode-dir', required=True)
    parser.add_argument('-s', '--split', choices=['val', 'test'], required=True)
    parser.add_argument('-o', '--output-file', required=True)
    args = parser.parse_args()

    results = defaultdict(OrderedDict)
    articles = {}
    for data_filepath in glob.iglob(os.path.join(args.data_dir, args.split, '*.json'), recursive=False):
        data_filename = os.path.basename(os.path.normpath(data_filepath))
        name, ext = os.path.splitext(data_filename)
        index = int(name)
        articles[index] = json.load(open(data_filepath, 'r', encoding='utf8'))['article']
        results[index]['article'] = '\n'.join(articles[index])

    summaries = defaultdict(OrderedDict)
    for model_folder in sorted(list(glob.iglob(os.path.join(args.decode_dir, '*'), recursive=False))):
        model_name = os.path.basename(os.path.normpath(model_folder))
        for dec_filepath in glob.iglob(os.path.join(model_folder, args.split, 'output', '*.dec'), recursive=False):
            dec_filename = os.path.basename(os.path.normpath(dec_filepath))
            name, ext = os.path.splitext(dec_filename)
            index = int(name)
            summaries[index][model_name] = [line.strip() for line in open(dec_filepath, 'r', encoding='utf8')]
            results[index][model_name] = '\n'.join(summaries[index][model_name])

    fieldnames = ['id'] + list(results[0].keys())
    with open(args.output_file, 'w', encoding='utf8') as out:
        csv_writer = csv.DictWriter(out, fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writeheader()
        for key in sorted(list(results.keys())):
            csv_writer.writerow({'id': key, **results[key]})

    output_file_name, _ = os.path.splitext(args.output_file)

    json_results = []
    for index in sorted(list(articles.keys())):
        json_results.append(OrderedDict([
            ('id', index),
            ('article', articles[index]),
            ('summaries', summaries[index]),
            ('readability_scores', {model: get_scores(texts) for model, texts in summaries[index].items()}),
        ]))
    with open(os.path.normpath(output_file_name) + '.json', 'w', encoding='utf8') as out:
        json.dump(json_results, out, indent=4, ensure_ascii=False)

    scores_results = []
    for model in summaries[0].keys():
        scores = [j['readability_scores'][model] for j in json_results]
        scores_results.append(OrderedDict(
            [('model', model)] +
            [(key, np.mean(list(s[key] for s in scores))) for key in scores[0].keys()]))

    with open(os.path.normpath(output_file_name) + '_readability_scores.csv', 'w', encoding='utf8') as out:
        csv_writer = csv.DictWriter(out, scores_results[0].keys(), quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writeheader()
        csv_writer.writerows(scores_results)


if __name__ == '__main__':
    main()
