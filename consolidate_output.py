import argparse
import csv
import glob
import json
import os
from collections import defaultdict, OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data-dir', required=True)
    parser.add_argument('-d', '--decode-dir', required=True)
    parser.add_argument('-s', '--split', choices=['val', 'test'], required=True)
    parser.add_argument('-o', '--output-file', required=True)
    args = parser.parse_args()

    results = defaultdict(OrderedDict)
    for data_filepath in glob.iglob(os.path.join(args.data_dir, args.split, '*.json'), recursive=False):
        data_filename = os.path.basename(os.path.normpath(data_filepath))
        name, ext = os.path.splitext(data_filename)
        results[int(name)]['article'] = '\n'.join(json.load(open(data_filepath, 'r', encoding='utf8'))['article'])

    for model_folder in sorted(list(glob.iglob(os.path.join(args.decode_dir, '*'), recursive=False))):
        model_name = os.path.basename(os.path.normpath(model_folder))
        for dec_filepath in glob.iglob(os.path.join(model_folder, 'output', '*.dec'), recursive=False):
            dec_filename = os.path.basename(os.path.normpath(dec_filepath))
            name, ext = os.path.splitext(dec_filename)
            results[int(name)][model_name] = ''.join(open(dec_filepath, 'r', encoding='utf8')).strip()

    fieldnames = ['id'] + list(results[0].keys())
    with open(args.output_file, 'w', encoding='utf8') as out:
        csv_writer = csv.DictWriter(out, fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writeheader()
        for key in sorted(list(results.keys())):
            csv_writer.writerow({'id': key, **results[key]})


if __name__ == '__main__':
    main()
