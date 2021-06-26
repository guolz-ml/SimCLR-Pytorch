import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        mask = torch.ones((2*batch_size, 2*batch_size),dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        self.mask = mask

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat((sim_ij, sim_ji), dim=0).reshape(2*self.batch_size, 1)
        negatives = similarity_matrix[self.mask].reshape(2*self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        labels = torch.zeros(2*self.batch_size).to(positives.device).long()
        loss = self.criterion(logits, labels) / (2*self.batch_size)
        return loss