import torch.nn.functional as F
from torch import nn


class ClassifierModule(nn.Module):
    def __init__(self, n_classes, dropout):
        super(ClassifierModule, self).__init__()
        n_hidden_neurons = 128
        self.embed_X_block = nn.Sequential(
            nn.Linear(in_features=124, out_features=n_hidden_neurons),
            nn.BatchNorm1d(num_features=n_hidden_neurons),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.mlp = nn.Linear(in_features=n_hidden_neurons, out_features=n_classes)

    def forward(self, x):
        embed_x = self.embed_X_block(x)
        logit_class = self.mlp(embed_x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        return p_class
