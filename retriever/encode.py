import copy
import queue
import threading
import torch
from tqdm import tqdm

from encoder import Encoder
from retriever.utils import save_encoding


def encode_texts(
    encoder: Encoder,
    texts: list[str],
    target: str,
    devices: list[str],
    bsize: int | None = None,
    show_progress: bool = True,
    output_path: str | None = None
) -> dict[str, torch.Tensor]:
    method = f"encode_{target}"
    if len(devices) > 1:
        encoding = _encode_parallel(
            encoder,
            texts,
            method,
            bsize,
            show_progress,
            devices
        )
    elif len(devices) == 1:
        encoder.model.to(devices[0])
        encoding = getattr(encoder, method)(texts, bsize, show_progress)
    else:
        raise ValueError("`devices` should not to be empty.")

    if output_path is not None:
        save_encoding(encoding, output_path)
    return encoding


def _encode_parallel(
    encoder: Encoder,
    texts: list[str],
    method: str,
    bsize: int | None,
    show_progress: bool,
    devices: list[str],
) -> dict[str, torch.Tensor]:
    if not texts:
        return {}

    effective_bsize = bsize if bsize is not None else len(texts)
    chunks = [texts[i : i + effective_bsize] for i in range(0, len(texts), effective_bsize)]
    total = len(chunks)

    work_q: queue.Queue = queue.Queue()
    for i, chunk in enumerate(chunks):
        work_q.put((i, chunk))

    k = min(len(devices), total)
    device_encoders = [
        Encoder(copy.deepcopy(encoder.model), encoder.tokenizer, dev)
        for dev in devices[:k]
    ]

    results: dict[int, dict] = {}
    pbar = tqdm(total=total, disable=not show_progress)
    lock = threading.Lock()

    def gpu_worker(enc: Encoder) -> None:
        encode_fn = getattr(enc, method)
        while True:
            try:
                chunk_idx, chunk = work_q.get_nowait()
            except queue.Empty:
                return
            enc_out = encode_fn(chunk, bsize=None, show_progress=False)
            with lock:
                results[chunk_idx] = enc_out
                pbar.update(1)

    threads = [threading.Thread(target=gpu_worker, args=(device_encoders[i],)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pbar.close()

    ordered = [results[i] for i in range(total)]
    return {key: torch.cat([r[key] for r in ordered]) for key in ordered[0]}
