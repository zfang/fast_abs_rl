from typing import List

import torch
from allennlp.modules.token_embedders import ElmoTokenEmbedder


class ElmoWordEmbedding(torch.nn.Module):
    """
    Compute a single layer of ELMo word representations.
    """

    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 vocab_to_cache: List[str],
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 requires_grad: bool = False,
                 projection_dim: int = None) -> None:
        super(ElmoWordEmbedding, self).__init__()

        self._elmo = ElmoTokenEmbedder(options_file=options_file,
                                       weight_file=weight_file,
                                       do_layer_norm=do_layer_norm,
                                       dropout=dropout,
                                       requires_grad=requires_grad,
                                       projection_dim=projection_dim,
                                       vocab_to_cache=vocab_to_cache)

        # self.weight is used to compute probabilities
        word_embedding_weight = self._elmo._elmo._elmo_lstm._word_embedding.weight
        self.weight: torch.nn.Parameter = torch.cat((word_embedding_weight, word_embedding_weight), dim=1)

    def get_output_dim(self):
        return self._elmo.get_output_dim()

    def forward(self, word_inputs: torch.Tensor) -> torch.Tensor:
        return self._elmo.forward(torch.zeros(word_inputs.shape), word_inputs)

    def cuda(self, device=None):
        super().cuda()
        self.weight = self.weight.to(next(self.parameters()).device)

        return self
