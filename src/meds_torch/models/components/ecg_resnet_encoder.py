import torch.nn as nn
import torchvision


class ECGResnetEncoder(nn.Module):
    """
    (B, 12, 5000)
    """

    def __init__(self, embedding_dim, kernel_size=7):
        super().__init__()
        self.encoder = torchvision.models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2, padding=2, bias=False)
        self.encoder.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)
