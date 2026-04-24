import torch


def insert_prefix_token_id(tensor: torch.Tensor, prefix_id: int) -> torch.Tensor:
    prefix_tensor = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)