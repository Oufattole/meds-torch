import dataclasses

import torch


@dataclasses.dataclass
class OutputBase:
    loss: torch.Tensor
