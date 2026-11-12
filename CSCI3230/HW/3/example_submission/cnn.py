"""Edit this CNN. The supplied baseline runs, but is NOT translation invariant."""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        width=int(config.get('width',16))
        # TODO: replace this architecture with your translation-invariant CNN.
        self.features=nn.Sequential(
            nn.Conv2d(1,width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(width,2*width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier=nn.Linear(2*width*7*7,10)

    def forward(self,x):
        return self.classifier(self.features(x).flatten(1))


def build_model(config):
    # Construction only: no data access, downloads, training or checkpoint loading.
    return CNN(config)
