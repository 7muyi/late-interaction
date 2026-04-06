import torch
import torch.nn as nn
from transformers import AutoModel
from scipy.cluster.hierarchy import linkage, fcluster

from utils import mv_score, maxsum


class TokenPooling(nn.Module):
    def __init__(self, pretrained_model: str, dim: int, pooling_factor: int) -> None:
        super().__init__()
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.proj = nn.Linear(self.llm.config.hidden_size, dim)

        self.pooling_factor = pooling_factor

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def hierarchical_pooling(self, encode_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tok_repr = encode_outputs["tok_repr"]
        tok_mask = encode_outputs["tok_mask"]

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
            #     vec_tensor = nn.functional.normalize(vec_tensor, p=2, dim=-1)
                
            #     padded_repr[b, idx, :] = vec_tensor
            #     padded_mask[b, idx] = 1

            """
            Optimized tensor version of the loop above.
            """
            labels_tensor = torch.from_numpy(labels).long().to(tok_repr.device) - 1

            cluster_sums = torch.zeros(target_k, D, device=tok_repr.device, dtype=tok_repr.dtype)
            cluster_sums.index_add_(0, labels_tensor, doc_vecs)

            counts = torch.bincount(labels_tensor).type_as(tok_repr).view(-1, 1)

            means = cluster_sums / counts
            means = nn.functional.normalize(means, p=2, dim=-1)

            padded_repr[b, :target_k] = means
            padded_mask[b, :target_k] = 1

        return {
            "tok_repr": padded_repr,
            "tok_mask": padded_mask
        }

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.llm(input_ids, attention_mask=attention_mask)[0]  # B, L, H

        tok_repr = self.proj(outputs)  # B, L, D
        tok_repr = torch.nn.functional.normalize(tok_repr, p=2, dim=-1)
        tok_repr = tok_repr * attention_mask.unsqueeze(-1)

        return {
            "tok_repr": tok_repr,
            "tok_mask": attention_mask
        }

    def encode_qry(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encode_outputs = self.encode(input_ids, attention_mask)
        return self.hierarchical_pooling(encode_outputs)

    @staticmethod
    def score(qry_repr: dict, doc_repr: dict, pairwise: bool = False) -> torch.Tensor:
        return maxsum(mv_score(qry_repr["tok_repr"], doc_repr["tok_repr"], pairwise))

    def forward(self, Q: tuple[torch.Tensor], D: tuple[torch.Tensor]) -> torch.Tensor:
        Q = self.encode_qry(*Q)
        D = self.encode_doc(*D)

        # default to in-negative sampling, so pairwise=False
        return TokenPooling.score(Q, D, False)