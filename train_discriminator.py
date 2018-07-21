import argparse
import os
from functools import lru_cache
from os.path import join

import torch
from cytoolz import compose, curry, concat
from toolz.sandbox import unzip
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.batcher import BucketedGenerater, pad_batch_tensorize, coll_fn_extract, \
    convert_batch_extract_ptr, prepro_fn_extract, tokenize
from data.data import CnnDmDataset
from decode_full_model import rerank
from decoding import Abstractor, BeamAbstractor
from model.cnn import ConvNet
from training import BasicPipeline, BasicTrainer, basic_validate, get_basic_grad_fn
from utils import UNK, PAD, get_elmo

BUCKET_SIZE = 6400
MAX_ABS_LEN = 30

HUMAN = 1
MACHINE = 0

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


@lru_cache(maxsize=None)
def get_abstractor(abs_dir, beam_search, cuda):
    if beam_search:
        return BeamAbstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        return Abstractor(abs_dir, MAX_ABS_LEN, cuda)


@curry
def abstract_callback(args, raw_article_sents):
    abstractor = get_abstractor(args.abs_dir, args.beam_search, args.cuda)

    with torch.no_grad():
        result = abstractor(raw_article_sents)
        if isinstance(abstractor, BeamAbstractor):
            result = rerank(result, [(0, len(raw_article_sents))])
        return [' '.join(dec) for dec in result]


class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, abs_results_path):
        super().__init__(split, DATA_DIR)
        self._abs_results_path = abs_results_path

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])

        if not extracts:
            return [], []

        abs_sents = abs_sents[:len(extracts)]
        with open(join(self._abs_results_path, '{}.dec'.format(i)), 'r', encoding='utf8') as f:
            abs_results = f.read().splitlines()

        texts = abs_sents + abs_results
        labels = [HUMAN] * len(abs_sents) + [MACHINE] * len(abs_results)
        return texts, labels


@curry
def batchify_fn(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    sources = pad_batch_tensorize(inputs=list(concat(source_lists)), pad=pad, cuda=cuda)
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources,)
    loss_args = (target,)
    return fw_args, loss_args


def build_batchers(args, word2id):
    decode_path = join(args.path, 'abs_decode')
    prepro = prepro_fn_extract(None, None)

    def sort_key(sample):
        texts, labels = sample
        return len(texts)

    batchify = compose(batchify_fn(PAD, cuda=args.cuda),
                       convert_batch_extract_ptr(UNK, word2id))

    train_loader = DataLoader(
        MatchDataset('train', decode_path),
        batch_size=BUCKET_SIZE,
        shuffle=not args.debug,
        num_workers=4 if args.cuda and not args.debug else 0,
        collate_fn=coll_fn_extract)

    train_batcher = BucketedGenerater(train_loader,
                                      prepro,
                                      sort_key,
                                      batchify,
                                      single_run=False,
                                      fork=not args.debug)

    val_loader = DataLoader(
        MatchDataset('val', decode_path),
        batch_size=BUCKET_SIZE,
        shuffle=False,
        num_workers=4 if args.cuda and not args.debug else 0,
        collate_fn=coll_fn_extract)

    val_batcher = BucketedGenerater(val_loader,
                                    prepro,
                                    sort_key,
                                    batchify,
                                    single_run=True,
                                    fork=not args.debug)

    return train_batcher, val_batcher


def decode(args, split):
    decode_path = join(args.path, 'abs_decode')
    os.makedirs(decode_path, exist_ok=True)

    dataset = CnnDmDataset(split, DATA_DIR)
    print('Generating abstracts for {} dataset'.format(split))
    for i in tqdm(range(len(dataset))):
        js_data = dataset[i]
        art_sents, extracts = (js_data['article'], js_data['extracted'])

        if not extracts:
            abs_results = []
        else:
            abs_results = abstract_callback(args, tokenize(None, (art_sents[i] for i in extracts)))

        with open(join(decode_path, '{}.dec'.format(i)), 'w', encoding='utf8') as f:
            f.write('\n'.join(abs_results))


def main(args):
    abstractor = get_abstractor(args.abs_dir, args.beam_search, args.cuda)
    for split in ('train', 'val'):
        decode(args, split)

    embedding = abstractor._net._decoder._embedding
    word2id = abstractor._word2id
    id2words = {i: w for w, i in word2id.items()}

    elmo = None
    if args.elmo:
        elmo = get_elmo(dropout=args.elmo_dropout,
                        vocab_to_cache=[id2words[i] for i in range(len(id2words))],
                        cuda=args.cuda)
        args.emb_dim = elmo.get_output_dim()

    meta = {
        'net': '{}_discriminator'.format('cnn'),
        'net_args': {
            'vocab_size': len(abstractor._word2id),
            'emb_dim': embedding.embedding_dim,
            'kernel_num': args.kernel_num,
            'kernel_sizes': args.kernel_sizes,
            'class_num': 2,
            'dropout': args.dropout,
            'max_norm': args.max_norm,
            'static': args.static,
        },
        'training_params': {
            'optimizer': ('adam', {'lr': args.lr}),
            'batch_size': args.batch,
            'clip_grad_norm': args.clip,
            'lr_decay': args.decay,
        }
    }

    net = ConvNet(**meta['net_args'])

    if elmo:
        net.set_elmo_embedding(elmo)
    else:
        net.set_embedding(embedding.weight)

    train_batcher, val_batcher = build_batchers(args, word2id)

    def criterion(logit, target):
        return F.cross_entropy(logit, target, reduce=False)

    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline('discriminator', net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the discriminator (CNN)'
    )
    parser.add_argument('--path', required=True, help='root of the discriminator model')
    parser.add_argument('--abs_dir', required=True, help='root of the abstractor model')

    parser.add_argument('--kernel_num', type=int, action='store', default=100,
                        help='the number of kernels for each size')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', action='store', default=[3, 4, 5],
                        help='kernels sizes')
    parser.add_argument('--static', action='store_true',
                        help='fix the embedding')
    parser.add_argument('--max_norm', type=float, default=3.0,
                        help='l2 constraint of parameters')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout')

    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--elmo', action='store_true',
                        help='augment embedding with elmo')
    parser.add_argument('--elmo-dropout', type=float, default=0,
                        help='the probability for elmo dropout')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
