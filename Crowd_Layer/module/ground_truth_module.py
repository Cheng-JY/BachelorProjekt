import torch.nn.functional as F
from torch import nn


class ClassifierModule(nn.Module):
    def __init__(self, n_classes, dropout):
        super(ClassifierModule, self).__init__()
        n_hidden_neurons_1 = 256
        n_hidden_neurons_2 = 128
        self.embed_X_block = nn.Sequential(
            nn.Linear(in_features=124, out_features=n_hidden_neurons_1),
            nn.BatchNorm1d(num_features=n_hidden_neurons_1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=n_hidden_neurons_1, out_features=n_hidden_neurons_2),
            nn.BatchNorm1d(num_features=n_hidden_neurons_2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.mlp = nn.Linear(in_features=n_hidden_neurons_2, out_features=n_classes)

    def forward(self, x):
        embed_x = self.embed_X_block(x)
        logit_class = self.mlp(embed_x)

        return logit_class
