import argparse
import json
import logging
import os
import sys
from time import time

import matplotlib
from tqdm import tqdm

from fast_abs_rl import preprocess, load_models, decode, get_embedding

matplotlib.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', required=True)
parser.add_argument('-o', '--output-file')
parser.add_argument('--relevance-scores-file')
parser.add_argument('--model-dir', required=True)
parser.add_argument('--beam-size', type=int, default=5)
parser.add_argument('--max-len', type=int, default=30)
parser.add_argument('--diverse', type=float, default=1)
parser.add_argument('-l', '--limit', type=int, default=0)
parser.add_argument('--lambda_', type=float, default=1)
parser.add_argument('--prepro', action='store_true')
parser.add_argument('--postpro', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--logging-level', choices=('critical',
                                                'fatal',
                                                'error',
                                                'warn',
                                                'info',
                                                'debug',
                                                ), default='info')
args = parser.parse_args()

logging_level = logging.getLevelName(args.logging_level.upper())

root = logging.getLogger()
root.setLevel(logging_level)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

with open(args.input_file, 'r', encoding='utf8') as f:
    raw_sentences = f.read().splitlines()

relevance_scores = None
if args.relevance_scores_file:
    with open(args.relevance_scores_file, 'r', encoding='utf8') as f:
        relevance_scores = list(map(float, f.read().splitlines()))

start = time()
extractor, abstractor = load_models(model_dir=args.model_dir,
                                    beam_size=args.beam_size,
                                    max_len=args.max_len)
logging.info('load models: {0:.3f}s'.format(time() - start))

if args.prepro:
    start = time()
    raw_sentences = preprocess(texts=raw_sentences,
                               limit=args.limit,
                               relevance_scores=relevance_scores,
                               embedding=get_embedding(extractor))
    logging.info('preprocess: {0:.3f}s'.format(time() - start))

start = time()
result = decode(raw_sentences,
                extractor,
                abstractor,
                beam_size=args.beam_size,
                diverse=args.diverse,
                postpro=args.postpro,
                debug=args.debug)

if args.debug:
    result, attns = result
    input_file_name, _ = os.path.splitext(args.input_file)
    model_dir = os.path.basename(args.model_dir)
    logging.info('creating heat maps...')
    for i in tqdm(range(len(attns))):
        attn = attns[i]
        plt.figure(figsize=(10, min(10, 2 * len(attn) // len(attn.columns))))
        sns.heatmap(attn, linewidths=0.1)
        plt.savefig("{}_{}_heatmap_{}.png".format(input_file_name, model_dir, i), dpi=300, bbox_inches='tight')
        plt.clf()
        attn.to_csv("{}_{}_heatmap_{}.csv".format(input_file_name, model_dir, i))

if args.output_file:
    out = open(args.output_file, 'w', encoding='utf8')
else:
    out = sys.stdout

json.dump(result, out, indent=4, ensure_ascii=False)
print()
