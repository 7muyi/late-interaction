import contextlib
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from base import BaseTokenizer, BaseConfig
from dataloader import get_dataloader
from utils import (set_seed, setup_ddp, cleanup_ddp,
                   get_scheduler, get_param_groups, log_metrics,
                   save_checkpoint, to_device, MixedPrecisionManager)


def run(
    model,
    tokenizer: BaseTokenizer,
    config: BaseConfig,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None
) -> None:
    rank, local_rank, world_size = setup_ddp(rank, local_rank, world_size)
    set_seed(42 + rank)

    device = torch.device(f"cuda:{local_rank}")

    dataloader = get_dataloader(
        config.dataset_path,
        tokenizer,
        config.bsize // world_size,
        rank,
        world_size
    )
    total_steps = (len(dataloader) + config.accumulation_steps - 1) // config.accumulation_steps * config.epoch

    model.train()
    model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank]
    )

    param_groups = get_param_groups(model.module, config.lr_backbone, config.lr_other)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_scheduler(
        optimizer,
        config.warmup,
        total_steps,
        config.lr_min_ratio
    )

    if rank == 0:
        os.makedirs(config.checkpoint_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.checkpoint_path, "log")) if rank == 0 else None

    amp = MixedPrecisionManager(config.amp)

    global_step = 0
    accumulation_loss = 0.0
    accumulation_count = 0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epoch):
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            Q, D = batch
            N = D[0].shape[0] // Q[0].shape[0]

            # Use no_sync() to skip gradient synchronization during accumulation
            is_accumulation_step = (batch_idx + 1) % config.accumulation_steps != 0 and (batch_idx + 1) != len(dataloader)
            context = model.no_sync() if is_accumulation_step else contextlib.nullcontext()

            with context:
                # calculate Query - Document score
                Q = to_device(Q, device)
                D = to_device(D, device)
                with amp.context():
                    scores = model(Q, D)

                    # Labels: each query's positive doc is at index i*N (first doc in each group)
                    labels = torch.arange(scores.shape[0], dtype=torch.long, device=device) * N
                    loss = torch.nn.functional.cross_entropy(scores / config.temperature, labels, reduction="mean")

                # Scale loss for gradient accumulation
                norm_loss = loss / config.accumulation_steps
                amp.backward(norm_loss)

                accumulation_loss += loss.item()
                accumulation_count += 1

            if not is_accumulation_step:
                amp.step(model, optimizer, scheduler, max_grad_norm=1.0)

                global_step += 1

                # Only the main process (rank 0) logs the information
                if rank == 0:
                    if global_step % config.log_interval == 0:
                        log_metrics(
                            writer,
                            {"loss": accumulation_loss / accumulation_count},
                            global_step
                        )
                        accumulation_loss = 0.0
                        accumulation_count = 0

        if rank == 0:
            ckpt_path = os.path.join(config.checkpoint_path, f"model_epoch{epoch+1}.pt")
            save_checkpoint(model.module, optimizer, scheduler, epoch + 1, global_step, config, ckpt_path)

    if writer:
        writer.close()

    cleanup_ddp()
