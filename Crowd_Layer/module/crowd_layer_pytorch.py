import torch
import torch.nn.functional as F
from torch import nn


class CrowdLayerPytorch(nn.Module):
    def __init__(
            self,
            n_classes,
            n_annotators,
            gt_net,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net

        # Setup crowd layer.
        self.annotator_layers = nn.ModuleList()
        for i in range(n_annotators):
            layer = nn.Linear(n_classes, n_classes, bias=False)
            layer.weight = nn.Parameter(torch.eye(n_classes) * 10)
            self.annotator_layers.append(layer)

    def forward(self, x):
        # Compute class-membership logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        # Compute logits per annotator.
        logits_annot = []
        for layer in self.annotator_layers:
            logits_annot.append(layer(p_class))
        logits_annot = torch.stack(logits_annot, dim=2)

        return p_class, logits_annot



