import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from scipy.cluster.hierarchy import linkage, fcluster

import scorer  # register scorer functions
from common.registry import registry
from models.base_model import BaseModel, BaseEncoder


@registry.register_model_name("tokenpooling")
class TokenPooling(BaseModel, BaseEncoder):
    def __init__(self, pretrained_model: str, dim: int, pooling_factor: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))

        self.pooling_factor = pooling_factor

        self._init_weights()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=0.1)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def hierarchical_pooling(self, encode_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tok_repr = encode_outputs["mv_repr"]
        tok_mask = encode_outputs["mv_mask"]

        B, L, D = tok_repr.size()

        valid_lens = tok_mask.sum(dim=1).long()
        valid_lens = torch.clamp(valid_lens, min=2)
        num_clusters_list = (valid_lens // self.pooling_factor) + 1
        max_k = num_clusters_list.max().item()

        padded_repr = torch.zeros(B, max_k, D, dtype=tok_repr.dtype, device=tok_repr.device)
        padded_mask = torch.zeros(B, max_k, dtype=tok_mask.dtype, device=tok_repr.device)

        for b in range(B):
            v_len = valid_lens[b].item()
            target_k = num_clusters_list[b].item()

            doc_vecs = tok_repr[b, :v_len, :].detach().cpu().numpy()

            Z = linkage(doc_vecs, method='ward')
            labels = fcluster(Z, t=target_k, criterion='maxclust')

            # actual_clusters = np.unique(labels)
            # for idx, cluster_id in enumerate(actual_clusters):
            #     cluster_indices = np.where(labels == cluster_id)
            #     cluster_mean = doc_vecs[cluster_indices].mean(axis=0)

            #     vec_tensor = torch.from_numpy(cluster_mean).to(tok_repr.device, dtype=tok_repr.dtype)
            #     vec_tensor = F.normalize(vec_tensor, p=2, dim=-1)

            #     padded_repr[b, idx, :] = vec_tensor
            #     padded_mask[b, idx] = 1

            """
            Optimized tensor version of the loop above.
            """
            labels_tensor = torch.from_numpy(labels).long().to(tok_repr.device) - 1

            cluster_sums = torch.zeros(target_k, D, device=tok_repr.device, dtype=tok_repr.dtype)
            cluster_sums.index_add_(0, labels_tensor, tok_repr[b, :v_len, :])

            counts = torch.bincount(labels_tensor).type_as(tok_repr).view(-1, 1)
            actual_k = counts.size(0)

            means = cluster_sums[:actual_k] / counts
            means = F.normalize(means, p=2, dim=-1)

            padded_repr[b, :actual_k] = means
            padded_mask[b, :actual_k] = 1

        return {
            "mv_repr": padded_repr,
            "mv_mask": padded_mask
        }

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tok_repr = self.proj(outputs)  # B, L, D
        tok_repr = F.normalize(tok_repr, p=2, dim=-1)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "mv_repr": tok_repr,
            "mv_mask": attention_mask
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encode_outputs = self.encode(input_ids, attention_mask)
        return self.hierarchical_pooling(encode_outputs)

    def score(self, qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return registry.get_scorer("maxsim_sum")(qry_repr, doc_repr, pairwise)

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return self.score(Q, D, False) / self.temperature

    @classmethod
    def from_config(cls, config):
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        pooling_factor = config.get("pooling_factor", 4)
        temperature = config.get("temperature", 1.0)

        return cls(pretrained_model, dim, pooling_factor, temperature)