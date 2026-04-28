import torch


class StrideTensor:
    def __init__(self, packed_tensor: torch.Tensor, lengths: torch.Tensor | list, device: str = "cuda") -> None:
        self.device = device
        # Lazy loading: load to the specified device only when indexed.
        self.tensor = packed_tensor
        self.dim = self.tensor.size(-1)
        self.lengths = lengths.long() if isinstance(lengths, torch.Tensor) else torch.LongTensor(lengths)
        # Calculate offsets: starting position of each sequence in packed tensor
        # offsets[i] = sum of all lengths before sequence i
        self.offsets = torch.cat((
            torch.zeros(1, dtype=torch.long, device=self.lengths.device),
            torch.cumsum(self.lengths, dim=0)[:-1]
        ))

        self.max_stride = self.lengths.max().item()
        if self.offsets[-1] + self.max_stride > self.tensor.size(0):
            padding = torch.zeros(self.max_stride, self.dim, dtype=self.tensor.dtype, device=self.tensor.device)
            self.tensor = torch.cat((self.tensor, padding))

    @classmethod
    def from_packed_tensor(cls, tensor: torch.Tensor, lengths: torch.Tensor | list, device: str = "cuda") -> "StrideTensor":
        return cls(tensor, lengths, device)

    @classmethod
    def from_padded_tensor(cls, tensor: torch.Tensor, mask: torch.Tensor, device: str = "cuda") -> "StrideTensor":
        return cls(tensor[mask.bool()], mask.sum(dim=-1), device)

    def lookup(self, id: int, num: int = 1) -> torch.Tensor:
        # Get lengths and offsets for the requested sequences
        lengths = self.lengths[id: id + num]
        offsets = self.offsets[id: id + num]

        tensor = _create_view(self.tensor, self.max_stride, self.dim)[offsets].to(self.device)
        # Create mask to zero out padding positions
        mask = _create_mask(lengths, self.max_stride).to(tensor.device)
        return tensor * mask.unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[self.offsets[index]: self.offsets[index] + self.lengths[index]]

def _create_view(tensor: torch.Tensor, stride: int, dim: int) -> torch.Tensor:
    size = (tensor.size(0) - stride + 1, stride, dim)
    strides = [dim, dim, 1]
    return torch.as_strided(tensor, size, strides)

def _create_mask(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    mask = torch.arange(stride, device=lengths.device)
    return mask.unsqueeze(0) < lengths.unsqueeze(-1)