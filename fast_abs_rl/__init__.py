from typing import Dict

import torch

from fast_abs_rl.decoding import load_models, decode, RLExtractor
from fast_abs_rl.preprocess import preprocess


class WordEmbedding:
    def __init__(self, word2id: Dict[str, int], embedding: torch.nn.Embedding):
        self.word2id = word2id
        self.embedding_data = embedding.weight.data.numpy()
        self.d_emb = self.embedding_data.shape[1]

    def __getitem__(self, item: str):
        return self.embedding_data[self.word2id[item]]

    def get(self, item: str, default):
        id = self.word2id.get(item)
        if id is None:
            if callable(default):
                return default()
            else:
                return default
        return self.embedding_data[id]


def get_embedding(extractor: RLExtractor):
    return WordEmbedding(word2id=extractor.word2id,
                         embedding=extractor.net._sent_enc._embedding)
