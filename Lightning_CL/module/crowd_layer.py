import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class CrowdLayerModule(L.LightningModule):
    def __init__(
            self,
            n_classes,
            n_annotators,
            gt_net,
            lr,
            weight_decay=1e-4
    ):
        super().__init__()
        L.seed_everything(42)
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net
        self.lr = lr
        self.weight_decay = weight_decay

        # Setup crowd layer.
        self.annotator_layers = nn.ModuleList()
        for i in range(n_annotators):
            layer = nn.Linear(n_classes, n_classes, bias=False)
            layer.weight = nn.Parameter(torch.eye(n_classes) * 10)
            self.annotator_layers.append(layer)

        self.save_hyperparameters()

    def forward(self, x):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.

        Returns
        -------
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.Tensor of shape (batch_size, n_classes, n_annotators)
            Annotation logits for each sample-annotator pair.
        """
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

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        _, logits_annot = self.forward(x)
        loss = F.cross_entropy(logits_annot, y, reduction="mean", ignore_index=-1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p_class, logits_annot = self.forward(x)
        y_pred = p_class.argmax(dim=-1)
        acc = (y_pred == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx):
        P_class, P_annot = self.forward(batch)
        P_annot = F.softmax(P_annot, dim=1)
        return P_class, P_annot



