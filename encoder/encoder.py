import torch
from tqdm import tqdm

from base import BaseTokenizer, BaseModel


class Encoder:
    def __init__(self, model: BaseModel, tokenizer: BaseTokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.model.eval()

    def _flatten(self, tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor[mask == 1], mask.sum(-1)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_fn,
        cpu: bool = False
    ) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        with torch.no_grad():
            output = encoder_fn(input_ids, attention_mask)

        encoding = {}
        if "cls_repr" in output:
            encoding["cls_repr"] = output["cls_repr"]

        if "mv_repr" in output:
            mv_repr, mv_lens = self._flatten(output["mv_repr"], output["mv_mask"])
            encoding["mv_repr"] = mv_repr
            encoding["mv_lens"] = mv_lens

        if cpu:
            return {k: v.cpu() for k, v in encoding.items()}
        return encoding

    def _encode_batch(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        show_progress: bool,
        encoder_fn,
    ) -> dict[str, torch.Tensor]:
        all_encoding = {}
        for batch in tqdm(batches, total=len(batches), disable=not show_progress):
            encoding = self._encode(
                *batch,
                encoder_fn=encoder_fn,
                cpu=True
            )

            for k, v in encoding.items():
                all_encoding.setdefault(k, []).append(v)

        return {k: torch.cat(v) for k, v in all_encoding.items()}

    def _build_batches(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        bsize: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(
            zip(
                torch.split(ids, bsize, dim=0),
                torch.split(mask, bsize, dim=0),
            )
        )

    def encode_qry(self, queries: list[str], bsize: int | None = None, show_progress: bool = True) -> dict[str, torch.Tensor]:
        ids, mask = self.tokenizer.tensorize_qry(queries)
        if bsize is not None:
            batches = self._build_batches(ids, mask, bsize)
            return self._encode_batch(
                batches,
                show_progress,
                self.model.encode_qry,
            )
        else:
            return self._encode(
                ids,
                mask,
                encoder_fn=self.model.encode_qry,
                cpu=True
            )

    def encode_doc(self, documents: list[str], bsize: int | None = None, show_progress: bool = True) -> dict[str, torch.Tensor]:
        ids, mask = self.tokenizer.tensorize_doc(documents)
        if bsize is not None:
            batches = self._build_batches(ids, mask, bsize)
            return self._encode_batch(
                batches,
                show_progress,
                self.model.encode_doc,
            )
        else:
            return self._encode(
                ids,
                mask,
                encoder_fn=self.model.encode_doc,
                cpu=True
            )