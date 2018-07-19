""" utility functions"""
import os
import re
from os.path import basename

import gensim
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import nn

PAD = 0
UNK = 1
START = 2
END = 3

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

ELMO = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, 1, dropout=0)


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


def make_embedding(id2word, w2v_file, initializer=None, augment_elmo=False):
    attrs = basename(w2v_file).split('.')  # word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1]) + int(augment_elmo) * ELMO.get_output_dim()
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
                tensor = torch.Tensor(w2v[word])
                if augment_elmo:
                    tensor = torch.cat(
                        (tensor, get_elmo_embedding([[word]]).squeeze()),
                        dim=0)

                embedding[i, :] = tensor

    return embedding, oovs


def make_elmo_embedding(id2word):
    words = [[id2word[i]] for i in range(len(id2word))]
    words[START] = ['<s>']
    words[END] = ['<\s>']

    return torch.nn.Parameter(get_elmo_embedding(words).squeeze())


def get_elmo_embedding(tokens):
    return ELMO(batch_to_ids(tokens))['elmo_representations'][0]
