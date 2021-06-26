import torch.nn as nn
import torchvision
from models.simclr_backbone import get_backbone


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_size, n_features):
        super().__init__()
        self.base_model = encoder
        self.n_features = n_features
        self.base_model.fc = Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_size, bias=False)
        )

    def forward(self, x):
        embedding = self.base_model(x)
        out = self.projector(embedding)
        return embedding, out
