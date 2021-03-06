""" train extractor (ML)"""
import argparse
import json
import os
import pickle as pkl
from os.path import join, exists

import torch
from cytoolz import compose
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from fast_abs_rl.data.batcher import BucketedGenerater
from fast_abs_rl.data.batcher import coll_fn_extract, prepro_fn_extract
from fast_abs_rl.data.batcher import convert_batch_extract_ff, batchify_fn_extract_ff
from fast_abs_rl.data.batcher import convert_batch_extract_ptr, batchify_fn_extract_ptr
from fast_abs_rl.data.data import CnnDmDataset
from fast_abs_rl.model.extract import ExtractSumm, PtrExtractSumm
from fast_abs_rl.model.util import sequence_loss
from fast_abs_rl.utils import PAD, UNK, get_elmo
from fast_abs_rl.utils import make_vocab, make_embedding
from training import BasicPipeline, BasicTrainer
from training import get_basic_grad_fn, basic_validate

BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class ExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        return art_sents, extracts


def build_batchers(net_type, word2id, cuda, debug):
    assert net_type in ['ff', 'rnn']
    prepro = prepro_fn_extract(args.max_word, args.max_sent)

    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)

    batchify_fn = (batchify_fn_extract_ff if net_type == 'ff'
                   else batchify_fn_extract_ptr)
    convert_batch = (convert_batch_extract_ff if net_type == 'ff'
                     else convert_batch_extract_ptr)
    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id))

    train_loader = DataLoader(
        ExtractDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        ExtractDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden,
                  lstm_hidden, lstm_layer, bidirectional):
    assert net_type in ['ff', 'rnn']
    net_args = {
        'vocab_size': vocab_size,
        'emb_dim': emb_dim,
        'conv_hidden': conv_hidden,
        'lstm_hidden': lstm_hidden,
        'lstm_layer': lstm_layer,
        'bidirectional': bidirectional
    }

    net = (ExtractSumm(**net_args) if net_type == 'ff'
           else PtrExtractSumm(**net_args))
    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['ff', 'rnn']
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay
    }

    if net_type == 'ff':
        criterion = lambda logit, target: F.binary_cross_entropy_with_logits(
            logit, target, reduce=False)
    else:
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)

        def criterion(logits, targets):
            return sequence_loss(logits, targets, ce, pad_idx=-1)

    return criterion, train_params


def main(args):
    assert args.net_type in ['ff', 'rnn']
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    id2words = {i: w for w, i in word2id.items()}

    elmo = None
    if args.elmo:
        elmo = get_elmo(dropout=args.elmo_dropout,
                        vocab_to_cache=[id2words[i] for i in range(len(id2words))],
                        cuda=args.cuda,
                        projection_dim=args.elmo_projection)
        args.emb_dim = elmo.get_output_dim()

    train_batcher, val_batcher = build_batchers(args.net_type,
                                                word2id,
                                                args.cuda,
                                                args.debug)

    # make net
    net, net_args = configure_net(args.net_type,
                                  len(word2id), args.emb_dim, args.conv_hidden,
                                  args.lstm_hidden, args.lstm_layer, args.bi)

    if elmo:
        net_args['elmo'] = {
            'dropout': args.elmo_dropout,
            'projection': args.elmo_projection,
        }
        net.set_elmo_embedding(elmo)
    elif args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding(
            id2words, args.w2v)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {
        'net': 'ml_{}_extractor'.format(args.net_type),
        'net_args': net_args,
        'traing_params': train_params
    }
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='rnn',
                        help='model type of the extractor (ff/rnn)')
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of lSTM')
    parser.add_argument('--lstm_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM Encoder')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
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
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--elmo', action='store_true',
                        help='augment embedding with elmo')
    parser.add_argument('--elmo-dropout', type=float, default=0,
                        help='the probability for elmo dropout')
    parser.add_argument('--elmo-projection', type=int, default=None,
                        help='projection dimension for elmo')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
