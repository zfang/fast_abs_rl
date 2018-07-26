""" full training (train rnn-ext + abs + RL) """
import argparse
import json
import os
import pickle as pkl
from itertools import cycle
from operator import itemgetter
from os.path import join

import torch
from cytoolz import identity
from toolz.sandbox.core import unzip
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from fast_abs_rl.data.batcher import tokenize
from fast_abs_rl.data.data import CnnDmDataset
from fast_abs_rl.decoding import Abstractor, RLExtractor, ArticleBatcher, BeamAbstractor
from fast_abs_rl.decoding import load_best_ckpt
from fast_abs_rl.model.extract import PtrExtractSumm
from fast_abs_rl.model.rl import ActorCritic
from fast_abs_rl.utils import get_elmo
from metric import compute_rouge_l, compute_rouge_n
from rl import A2CPipeline, set_shift_reward_mean
from rl import get_grad_fn
from training import BasicTrainer

MAX_ABS_LEN = 30

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""

    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents


def load_ext_net(ext_dir):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] == 'ml_rnn_extractor'
    ext_ckpt = load_best_ckpt(ext_dir)
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    elmo = None
    if 'elmo' in ext_args:
        elmo_args = ext_args['elmo']
        vocab_to_cache = [w for w, i in sorted(list(vocab.items()), key=itemgetter(1))]
        elmo = get_elmo(dropout=elmo_args.get('dropout', 0),
                        vocab_to_cache=vocab_to_cache)
        del ext_args['elmo']
    ext = PtrExtractSumm(**ext_args)
    if elmo is not None:
        ext.set_elmo_embedding(elmo)
    ext.load_state_dict(ext_ckpt)
    return ext, vocab


def configure_pretrained_net(args):
    """ load pretrained sub-modules and build the actor-critic network"""
    abs_dir = os.path.join(args.pretrained_dir, 'abstractor/')
    if args.beam_search:
        abstractor = BeamAbstractor(abs_dir, MAX_ABS_LEN, args.cuda)
    else:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, args.cuda)

    ext_dir = args.pretrained_dir
    extractor = RLExtractor(ext_dir, cuda=args.cuda)
    agent = extractor.net
    agent_vocab = extractor.word2id

    net_args = {
        'abstractor': (None if abs_dir is None else json.load(open(join(abs_dir, 'meta.json')))),
        'extractor': json.load(open(join(ext_dir, 'meta.json')))['net_args']['extractor']
    }

    return agent, agent_vocab, abstractor, net_args


def configure_net(args):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if args.abs_dir is not None:
        if args.beam_search:
            abstractor = BeamAbstractor(args.abs_dir, MAX_ABS_LEN, args.cuda)
        else:
            abstractor = Abstractor(args.abs_dir, MAX_ABS_LEN, args.cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(args.ext_dir)
    agent = ActorCritic(extractor._sent_enc,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, args.cuda))
    if args.cuda:
        agent = agent.cuda()

    net_args = {
        'abstractor': json.load(open(join(args.abs_dir, 'meta.json'))),
        'extractor': json.load(open(join(args.ext_dir, 'meta.json')))
    }

    return agent, agent_vocab, abstractor, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay,
        'gamma': gamma,
        'reward': reward,
        'stop_coeff': stop_coeff,
        'stop_reward': stop_reward
    }

    return train_params


def build_batchers(batch_size):
    def coll(batch):
        art_batch, abs_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents

    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader


def train(args):
    os.makedirs(args.path, exist_ok=True)
    set_shift_reward_mean(not args.no_reward_mean_shift)

    # make net
    if args.pretrained_dir:
        agent, agent_vocab, abstractor, net_args = configure_pretrained_net(args)
    else:
        agent, agent_vocab, abstractor, net_args = configure_net(args)

    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    train_batcher, val_batcher = build_batchers(args.batch)
    # TODO different reward
    reward_fn = compute_rouge_l
    stop_reward_fn = compute_rouge_n(n=1)

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {'state_dict': load_best_ckpt(args.abs_dir)}
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'), exist_ok=True)
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f)
    # save configuration
    meta = {
        'net': 'rnn-ext_abs_rl',
        'net_args': net_args,
        'train_params': train_params
    }
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    pipeline = A2CPipeline(meta['net'], agent, abstractor,
                           train_batcher, val_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma,
                           stop_reward_fn, args.stop)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model')
    parser.add_argument('--pretrained_dir', action='store',
                        help='root of the pretrained model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')

    # training options
    parser.add_argument('--reward', action='store', default='rouge-l',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=3,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--no-reward-mean-shift', action='store_true',
                        help='use min-max scaling instead of normalization on reward')
    parser.add_argument('--beam-search', action='store_true',
                        help='use beam search on abstractor')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    train(args)
