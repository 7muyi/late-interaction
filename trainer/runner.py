import contextlib
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .utils import (get_param_groups, MixedPrecisionManager, is_main_process, to_device, log_metrics, save_checkpoint)
from common.registry import registry


def run(
    model,
    dataloader,
    config
) -> None:
    device = torch.device(f"cuda:{config.local_rank}")

    model.train()
    model.to(device)
    model = DDP(
        model,
        device_ids=[config.local_rank],
        find_unused_parameters=True
    )

    param_groups = get_param_groups(model.module, config.lr_backbone, config.lr_other)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = registry.get_lr_scheduler_func(config.lr_sched)(
        optimizer=optimizer,
        num_warmup_steps=config.warmup,
        num_training_steps=(len(dataloader) + config.accumulation_steps - 1) // config.accumulation_steps * config.epoch,
        min_lr_ratio=config.get("min_lr_ratio", 0)
    )

    if is_main_process():
        os.makedirs(config.checkpoint_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.checkpoint_path, "log")) if is_main_process() else None

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
                    # temperature = config.temperature if config.get("temperature", None) is not None else model.module.temperature
                    loss = torch.nn.functional.cross_entropy(scores, labels, reduction="mean")

                # Scale loss for gradient accumulation
                norm_loss = loss / config.accumulation_steps
                amp.backward(norm_loss)

                accumulation_loss += loss.item()
                accumulation_count += 1

            if not is_accumulation_step:
                amp.step(model, optimizer, scheduler, max_grad_norm=1.0)

                global_step += 1

                # Only the main process (rank 0) logs the information
                if is_main_process():
                    if global_step % config.log_interval == 0:
                        log_metrics(
                            writer,
                            {"loss": accumulation_loss / accumulation_count},
                            global_step
                        )
                        accumulation_loss = 0.0
                        accumulation_count = 0

        if is_main_process():
            ckpt_path = os.path.join(config.checkpoint_path, f"model_epoch{epoch+1}.pt")
            save_checkpoint(
                path=ckpt_path,
                model_state_dict=model.module.state_dict(),
                epoch=epoch + 1,
                global_step=global_step
            )

    if writer:
        writer.close()
