import operator
import os
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.decomposition import TruncatedSVD
from wordfreq import word_frequency

PAD = 0
UNK = 1

WEIGHT_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'enwiki_vocab_min200.txt')


def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    return TruncatedSVD(n_components=npc, n_iter=7, random_state=0).fit(X).components_


def remove_pc(X, npc):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, rmpc=1):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if rmpc > 0:
        emb = remove_pc(emb, rmpc)
    return emb


def get_word_weights(words, a=1e-3):
    a = min(a, 1)
    return {word: a / (a + word_frequency(word, 'en')) for word in words}


def random_word_emb(dim):
    bound = 1e-2
    return np.random.uniform(-bound, bound, dim).astype('float32')


def get_word_embeddings(embedding, sentences):
    vocab = {token for sent in sentences for token in sent}
    word2id = defaultdict(lambda: UNK)
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    for i, w in enumerate(vocab, len(word2id)):
        word2id[w] = i

    id2word = np.asarray([word for word, _ in sorted(word2id.items(), key=operator.itemgetter(1))])

    word_emb = np.asarray([
        embedding.get(
            id2word[i],
            default=partial(random_word_emb, embedding.d_emb))
        for i in range(len(word2id))
    ])

    return word2id, id2word, word_emb


def prepare_data(list_of_seqs, dtype='float32'):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype(dtype)
    x_mask = np.zeros((n_samples, max_len)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def get_sif_embeddings(embedding, sentences):
    word2id, id2word, word_emb = get_word_embeddings(embedding, sentences)
    word_weights = get_word_weights(word2id.keys())
    x, _ = prepare_data(np.asarray([[word2id[word] for word in sent] for sent in sentences]), dtype='int32')
    w, _ = prepare_data(np.asarray([[word_weights[word] for word in sent] for sent in sentences]), dtype='float32')
    return SIF_embedding(We=word_emb, x=x, w=w)
