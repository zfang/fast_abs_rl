""" make reference text files needed for ROUGE evaluation """
import argparse
import json
import os
from datetime import timedelta
from os.path import join
from time import time

from fast_abs_rl.decoding import make_html_safe
from fast_abs_rl.utils import count_data

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def dump(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    dump_dir = join(DATA_DIR, 'refs', split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100 * i / n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['abstract']
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def main():
    for split in ['val', 'test']:  # evaluation of train data takes too long
        if not os.path.exists(join(DATA_DIR, split)):
            continue
        os.makedirs(join(DATA_DIR, 'refs', split), exist_ok=True)
        dump(split)


if __name__ == '__main__':
    main()
