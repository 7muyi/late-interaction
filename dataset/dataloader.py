import json

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from tokenizers import BaseTokenizer


class IRDataset(Dataset):
    def __init__(self, triplet_path: str, queries_path: str, corpus_path: str):
        self.queries_path = queries_path
        self.corpus_path = corpus_path

        # Build {id: byte_offset} dicts by scanning JSONL files once.
        # Text is NOT loaded into memory — only offsets are stored.
        self.query_offsets: dict[str, int] = {}
        with open(queries_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                item = json.loads(line)
                self.query_offsets[item["id"]] = offset

        self.corpus_offsets: dict[str, int] = {}
        with open(corpus_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                item = json.loads(line)
                self.corpus_offsets[item["id"]] = offset

        # Load triplet file fully into memory — only string IDs, negligible size.
        self.triplets: list[tuple[str, list[str]]] = []
        with open(triplet_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                self.triplets.append((parts[0], parts[1:]))

    def _read_text(self, file, offset: int) -> str:
        file.seek(offset)
        return json.loads(file.readline())["text"]

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        # Lazy-initialize per-worker file handles (safe with DataLoader forking).
        if not hasattr(self, "_qry_file"):
            self._qry_file = open(self.queries_path, "rb")
            self._corpus_file = open(self.corpus_path, "rb")

        qry_id, doc_ids = self.triplets[idx]
        # doc_ids[0] is the positive; rest are negatives.
        qry = self._read_text(self._qry_file, self.query_offsets[qry_id])
        docs = [self._read_text(self._corpus_file, self.corpus_offsets[did]) for did in doc_ids]
        return qry, docs


def collate_fn(batch: list[tuple[str, list[str]]], tokenizer: BaseTokenizer):
    queries, docs_list = zip(*batch)

    Q = tokenizer.tensorize_qry(list(queries))
    D = tokenizer.tensorize_doc([doc for docs in docs_list for doc in docs])

    return Q, D


def get_dataloader(triplet_path: str, queries_path: str, corpus_path: str,
                   tokenizer: BaseTokenizer, bsize: int,
                   rank: int, world_size: int, num_workers: int = 4) -> DataLoader:
    dataset = IRDataset(triplet_path, queries_path, corpus_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=bsize,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True
    )
