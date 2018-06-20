import argparse
import csv
import glob
import os
import re
from collections import OrderedDict

from tqdm import tqdm

rouge_parse_mapping = [
    {'name': 'ROUGE-1 Average_R', 'line_no': 1},
    {'name': 'ROUGE-1 Average_P', 'line_no': 2},
    {'name': 'ROUGE-1 Average_F', 'line_no': 3},
    {'name': 'ROUGE-2 Average_R', 'line_no': 5},
    {'name': 'ROUGE-2 Average_P', 'line_no': 6},
    {'name': 'ROUGE-2 Average_F', 'line_no': 7},
    {'name': 'ROUGE-L Average_R', 'line_no': 9},
    {'name': 'ROUGE-L Average_P', 'line_no': 10},
    {'name': 'ROUGE-L Average_F', 'line_no': 11},
]

meteor_parse_mapping = [
    {'name': 'Precision', 'line_no': -7},
    {'name': 'Recall', 'line_no': -6},
    {'name': 'f1', 'line_no': -5},
    {'name': 'fMean', 'line_no': -4},
    {'name': 'Fragmentation penalty', 'line_no': -3},
    {'name': 'Final score', 'line_no': -1},
]


def parse_file(file, parse_mapping):
    lines = list(open(file, 'r', encoding='utf8'))
    result = OrderedDict()
    for m in parse_mapping:
        result[m['name']] = float(re.search(
            '{}:\s+(?P<value>[-+]?\d*\.\d+|\d+)'.format(m['name']),
            lines[m['line_no']].strip()).group('value'))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-s', '--split', choices=['val', 'test'], required=True)
    parser.add_argument('-o', '--output-file')
    args = parser.parse_args()
    args.output_file = args.output_file or 'results_{}.csv'.format(args.split)

    all_folders = sorted(list(glob.iglob(os.path.join(args.input_dir, '*'), recursive=False)))
    with open(args.output_file, 'w', encoding='utf8') as out:
        fieldnames = ['model'] + \
                     [m['name'] for m in rouge_parse_mapping] + \
                     ['METEOR ' + m['name'] for m in meteor_parse_mapping]
        csv_writer = csv.DictWriter(out, fieldnames)
        csv_writer.writeheader()
        for i in tqdm(range(len(all_folders))):
            folder = os.path.join(all_folders[i], args.split)
            rouge_file = os.path.join(folder, 'rouge.txt')
            meteor_file = os.path.join(folder, 'meteor.txt')
            if not (os.path.exists(rouge_file) and os.path.exists(meteor_file)):
                for f in (rouge_file, meteor_file):
                    if not os.path.exists(f):
                        print('Could not find {}'.format(f))
                continue
            rouge_result = parse_file(rouge_file, rouge_parse_mapping)
            meteor_result = parse_file(meteor_file, meteor_parse_mapping)
            result = {'model': os.path.basename(os.path.normpath(all_folders[i]))}
            result.update(rouge_result)
            for k, v in meteor_result.items():
                result['METEOR ' + k] = v
            csv_writer.writerow(result)


if __name__ == '__main__':
    main()
