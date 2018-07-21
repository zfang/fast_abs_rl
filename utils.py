""" utility functions"""
import os
import re
from os.path import basename

import gensim
import torch
from torch import nn

from model.elmo import ElmoWordEmbedding

PAD = 0
UNK = 1
START = 2
END = 3

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    names = os.listdir(path)
    n_data = len(list(filter(lambda name: bool(matcher.match(name)), names)))
    return n_data


def make_vocab(wc, vocab_size):
    word2id = {
        '<pad>': PAD,
        '<unk>': UNK,
        '<start>': START,
        '<end>': END,
    }
    for i, (w, _) in enumerate(wc.most_common(vocab_size), len(word2id)):
        word2id[w] = i
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            word = None
            if i == START:
                word = '<s>'
            elif i == END:
                word = r'<\s>'
            elif id2word[i] in w2v:
                word = id2word[i]
            else:
                oovs.append(i)

            if word is not None:
                embedding[i, :] = torch.Tensor(w2v[word])

    return embedding, oovs


def get_elmo(dropout=0.5, requires_grad=False, vocab_to_cache=None, cuda=True):
    elmo = ElmoWordEmbedding(options_file=ELMO_OPTIONS_FILE,
                             weight_file=ELMO_WEIGHT_FILE,
                             dropout=dropout,
                             requires_grad=requires_grad,
                             vocab_to_cache=vocab_to_cache)
    if cuda:
        elmo = elmo.cuda()

    return elmo
