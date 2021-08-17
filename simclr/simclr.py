import torch.nn as nn
import torchvision
from models.simclr_backbone import get_backbone


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_size, n_features):
        super().__init__()
        self.enc = encoder
        self.n_features = n_features

        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        #Non-Linear Projection with a ReLU
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_size)
        )

    def forward(self, x):
        embedding = self.enc(x)
        out = self.projector(embedding)
        return embedding, out
