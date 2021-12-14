import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 5),
            nn.BatchNorm1d(5),
            nn.ELU(),

            nn.Linear(5, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Linear(4, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Linear(4, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),

            nn.Linear(5, 8),
        )

    def forward(self, X):
        X = self.encoder(X)
        out = self.decoder(X)
        return out
