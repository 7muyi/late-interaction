import os
from typing import Any

import torch
import torch.distributed as dist
from contextlib import nullcontext

from base import *


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sv_score(qry_repr: torch.Tensor, doc_repr: torch.Tensor, pairwise: bool = False) -> torch.Tensor:
    if pairwise:
        N = doc_repr.shape[0] // qry_repr.shape[0]
        doc_repr = doc_repr.view(-1, N, doc_repr.shape[-1])  # Q, N, d
        scores = (qry_repr.unsqueeze(1) * doc_repr).sum(-1)  # Q, N
    else:
        scores = qry_repr @ doc_repr.transpose(0, 1)  # Q, D
    return scores


def mv_score(qry_repr: torch.Tensor, doc_repr: torch.Tensor, pairwise: bool = False) -> torch.Tensor:
    if pairwise:
        N = doc_repr.shape[0] // qry_repr.shape[0]

        Q = qry_repr.unsqueeze(1)  # Q, 1, Lq, d
        D = doc_repr.view(-1, N, *doc_repr.shape[-2:])  # Q, N, Ld, d

        scores = torch.einsum("bild,bnjd->bnij", Q, D)  # Q, N, Lq, Ld
    else:  # inbatch negative sampling
        scores = torch.einsum("qik,djk->qdij", qry_repr, doc_repr)  # Q, D, Lq, Ld
    return scores


def maxsum(mv_scores: torch.Tensor) -> torch.Tensor:
    return mv_scores.max(-1).values.sum(-1)


def insert_prefix_token_id(tensor: torch.Tensor, prefix_id: int) -> torch.Tensor:
    prefix_tensor = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


def setup_ddp(rank: int | None, local_rank: int | None, world_size: int | None) -> tuple[int, int, int]:
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_scheduler(optimizer, warmup_steps, total_steps, min_ratio):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_ratio, 1.0 - progress * (1.0 - min_ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_param_groups(model, lr_backbone: float, lr_other: float):
    llm_params = list(model.llm.parameters())
    llm_ids = {id(p) for p in llm_params}
    other_params = [p for p in model.parameters() if id(p) not in llm_ids]
    return [
        {"params": llm_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr_other}
    ]


def log_metrics(writer, records: dict, step: int) -> None:
    for k, v in records.items():
        val = v.item() if hasattr(v, "item") else v
        prefix = "loss" if "loss" in k else "train"
        writer.add_scalar(f"{prefix}/{k}", val, step)


def save_checkpoint(path: str, model_state_dict, **kwargs) -> None:
    payload = {"model_state_dict": model_state_dict, **kwargs}
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict[str, Any]:
    ckpt = torch.load(path, weights_only=False)
    return ckpt


def to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch)


class MixedPrecisionManager():
    def __init__(self, activated) -> None:
        self.activated = activated

        if self.activated:
            self.scaler = torch.amp.GradScaler(device="cuda")

    def context(self):
        return torch.amp.autocast("cuda") if self.activated else nullcontext()

    def backward(self, loss) -> None:
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None, max_grad_norm: float = 1.0) -> None:
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), max_grad_norm)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)