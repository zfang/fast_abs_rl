import torch
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    """
    Convolutional word-level sentence classifier
     w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """

    def __init__(self, vocab_size, emb_dim, kernel_num, kernel_sizes, class_num, dropout, max_norm, static):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._embedding.weight.requires_grad = not static
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, kernel_num, i) for i in kernel_sizes])
        self._dropout = nn.Dropout(dropout)
        self._max_norm = max_norm
        self._fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
        self._fc1.weight.data.normal_().mul_(0.01)
        self._fc1.bias.data.zero_()

    def forward(self, input_):
        if self._max_norm is not None and self.training:
            with torch.no_grad():
                self._fc1.weight.data.renorm_(2, 0, self._max_norm)

        emb_input = self._embedding(input_)
        conv_in = emb_input.transpose(1, 2)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)

        return self._fc1(self._dropout(output))

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

    def set_elmo_embedding(self, embedding):
        self._embedding = embedding
