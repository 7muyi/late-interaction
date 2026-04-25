import os

import numpy as np
import random
import torch
import torch.distributed as dist
from contextlib import nullcontext


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return dist.get_rank() == 0


def setup_ddp(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=args.world_size,
        rank=args.rank
    )


def cleanup_ddp() -> None:
    dist.destroy_process_group()


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

    def step(self, model, optimizer, scheduler=None, max_grad_norm: float = 1.0) -> None:
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)


def get_param_groups(model, lr_backbone: float, lr_other: float):
    llm_params = list(model.llm.parameters())
    llm_ids = {id(p) for p in llm_params}
    other_params = [p for p in model.parameters() if id(p) not in llm_ids]
    return [
        {"params": llm_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr_other},
    ]


def to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch)


def log_metrics(writer, records: dict, step: int) -> None:
    for k, v in records.items():
        val = v.item() if hasattr(v, "item") else v
        prefix = "loss" if "loss" in k else "train"
        writer.add_scalar(f"{prefix}/{k}", val, step)


def save_checkpoint(path: str, model_state_dict, **kwargs) -> None:
    payload = {"model_state_dict": model_state_dict, **kwargs}
    torch.save(payload, path)