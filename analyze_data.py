import argparse
import csv
import glob
import json
import os
import string
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats
from tqdm import tqdm


def get_data(data_dir):
    def json_load_file(**kwargs):
        return json.load(open(**kwargs))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for split in ('train', 'val', 'test'):
            split_dir = os.path.join(data_dir, split)
            if os.path.exists(split_dir):
                for file_path in glob.iglob(os.path.join(split_dir, '*.json')):
                    futures.append(executor.submit(json_load_file, file=file_path, mode='r', encoding='utf8'))

        for i in tqdm(range(len(futures))):
            futures[i].result()

        return [f.result() for f in futures]


def get_sentence_count(data):
    return np.asarray([len(d['article']) for d in data])


def get_token_per_sentence_count(data):
    return np.asarray([len(t) for d in data for t in d['tokens']])


def get_token_per_article_count(data):
    return np.asarray([sum(len(t) for t in d['tokens']) for d in data])


def get_stats(counts):
    mode = stats.mode(counts)
    return dict(mean=np.mean(counts),
                median=np.median(counts),
                min=np.min(counts),
                max=np.max(counts),
                p90=np.percentile(counts, 90),
                p95=np.percentile(counts, 95),
                p99=np.percentile(counts, 99),
                std=np.std(counts),
                mode=dict(value=mode[0][0], count=mode[1][0]))


def tokenize(data):
    def func(d):
        d['tokens'] = [[t for t in sent.split() if t not in string.punctuation] for sent in d['article']]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(func, d) for d in data]

        for i in tqdm(range(len(futures))):
            futures[i].result()


def main():
    parser = argparse.ArgumentParser(description='Analyze data')
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-o', '--output-file', default='data_analysis.csv')
    args = parser.parse_args()

    print('Loading data...')
    data = get_data(args.data_dir)

    print('Tokenizing data...')
    tokenize(data)

    stats_mapping = [
        ('sentences', get_sentence_count),
        ('tokens_per_article', get_token_per_article_count),
        ('tokens_per_sentence', get_token_per_sentence_count),
    ]

    fieldnames = ['name', 'mean', 'median', 'min', 'max', 'p90', 'p95', 'p99', 'std', 'mode']
    with open(args.output_file, 'w', encoding='utf8') as out:
        csv_writer = csv.DictWriter(out, fieldnames)
        csv_writer.writeheader()
        for name, func in stats_mapping:
            print('Computing stats for {}...'.format(name))
            csv_writer.writerow({'name': name, **get_stats(func(data))})


if __name__ == '__main__':
    main()
