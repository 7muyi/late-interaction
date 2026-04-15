import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from scipy.cluster.hierarchy import linkage, fcluster

from common.registry import registry
from .base_model import BaseModel, BaseEncoder
from .utils import mv_score


@registry.register_model_name("tokenpooling")
class TokenPooling(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, pooling_factor: int) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)

        self.pooling_factor = pooling_factor

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        Q = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        Q = self.proj(Q)  # B, L, D
        Q = F.normalize(Q, p=2, dim=-1)
        Q = Q * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": Q,
            "mv_mask": attention_mask
        }

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        D = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        D = self.proj(D)  # B, L, D
        B, L, d_dim = D.size()

        valid_lens = attention_mask.sum(dim=1).long()
        valid_lens = torch.clamp(valid_lens, min=2)
        num_clusters_list = (valid_lens // self.pooling_factor) + 1
        max_k = num_clusters_list.max().item()

        padded_repr = torch.zeros(B, max_k, d_dim, dtype=D.dtype, device=D.device)
        padded_mask = torch.zeros(B, max_k, dtype=attention_mask.dtype, device=D.device)

        for b in range(B):
            v_len = valid_lens[b].item()
            target_k = num_clusters_list[b].item()

            doc_vecs = D[b, :v_len].detach().cpu().numpy()

            Z = linkage(doc_vecs, method='ward')
            labels = fcluster(Z, t=target_k, criterion='maxclust')

            labels_tensor = torch.from_numpy(labels).long().to(D.device) - 1

            cluster_sums = torch.zeros(target_k, d_dim, device=D.device, dtype=D.dtype)
            cluster_sums.index_add_(0, labels_tensor, D[b, :v_len, :])

            counts = torch.bincount(labels_tensor, minlength=target_k).type_as(D).view(-1, 1)

            means = cluster_sums / counts
            means = F.normalize(means, p=2, dim=-1)

            padded_repr[b, :target_k] = means
            padded_mask[b, :target_k] = 1

        return {
            "mv_repr": padded_repr,
            "mv_mask": padded_mask
        }

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        P = mv_score(qry_repr["mv_repr"], doc_repr["mv_repr"], pairwise)
        scores = P.max(dim=-1).values.sum(-1)

        if "mv_mask" in qry_repr:
            scores = scores / qry_repr["mv_mask"].sum(-1, keepdim=True)

        return scores

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return TokenPooling.score(Q, D, False)

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pooling_factor = config.get("pooling_factor", 4)

        return cls(pretrained_model, dim, pooling_factor)