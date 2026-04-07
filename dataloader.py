from torch.utils.data import Dataset, DataLoader, DistributedSampler

from base.tokenizer import BaseTokenizer


class TSVDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.offsets = []

        with open(file_path, "rb") as f:
            offset = 0
            self.offsets.append(offset)
            for line in f:
                offset += len(line)
                self.offsets.append(offset)
            self.offsets.pop()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        with open(self.file_path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8").rstrip("\n\r")

        qry, *docs = line.split("\t")
        return qry, docs


def collate_fn(batch: list[tuple[str, list[str]]], tokenizer: BaseTokenizer):
    queries, docs_list = zip(*batch)

    Q = tokenizer.tensorize_qry(list(queries))
    D = tokenizer.tensorize_doc([doc for docs in docs_list for doc in docs])

    return Q, D

def get_dataloader(file_path: str, tokenizer: BaseTokenizer, bsize: int,
                   rank: int, world_size: int, num_workers: int = 4) -> DataLoader:
    dataset = TSVDataset(file_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=bsize,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True
    )