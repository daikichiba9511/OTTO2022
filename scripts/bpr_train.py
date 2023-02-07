from __future__ import annotations

import torch
import torch.nn as nn


class BPR(nn.Module):
    """Bayesian Personalized Ranking model

    Reference:
    1.https://github.com/guoyang9/BPR-pytorch/blob/master/model.py
    """

    def __init__(self, user_num: int, item_num: int, factor_num: int) -> None:
        self._embedding_user = nn.Embedding(user_num, factor_num)
        self._embedding_item = nn.Embedding(item_num, factor_num)

    def _initialize_embed_weights(self, user_std: float = 0.01, item_std: float = 0.01) -> None:
        nn.init.normal_(self._embedding_user.weight, std=user_std)
        nn.init.normal_(self._embedding_item.weight, std=item_std)

    def forward(
        self, user: torch.Tensor, item_i: torch.Tensor, item_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_embed_vec = self._embedding_user(user)
        item_i_embed_vec = self._embedding_item(item_i)
        item_j_embed_vec = self._embedding_item(item_j)

        prediction_i = (user_embed_vec * item_i_embed_vec).sum(dim=-1)
        prediction_j = (user_embed_vec * item_j_embed_vec).sum(dim=-1)
        return prediction_i, prediction_j


class BPRDataset(torch.utils.data.Dataset):
    """Dataset for training of BPR model"""

    def __init__(self, config) -> None:
        self._config = config

    def __getitem__(self, index: int) -> tuple:
        ...
