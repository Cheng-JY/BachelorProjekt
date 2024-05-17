import numpy as np
import torch
import torch.nn.functional as F
from skactiveml.base import AnnotatorModelMixin
from skorch import NeuralNet
from skorch.dataset import unpack_data
from torch import nn
from torch.nn import CrossEntropyLoss
from skactiveml.utils import unlabeled_indices

from .skorch_classifier import SkorchClassifier


class CrowdLayerClassifier(SkorchClassifier, AnnotatorModelMixin):
    def __init__(self, *args, **kwargs):
        super(CrowdLayerClassifier, self).__init__(
            module=CrowdLayerModule,
            *args,
            criterion=CrossEntropyLoss(),
            criterion__reduction="mean",
            criterion__ignore_index=-1,
            **kwargs,
        )

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # unpack the tuple from the forward function
        p_class, logits_annot = y_pred
        loss = NeuralNet.get_loss(self, logits_annot, y_true, *args, **kwargs)
        return loss

    def fit(self, X, y, **fit_params):
        # utils _label_encode.fit_transform
        # _valid_data() in Skactivml
        is_unlbld = unlabeled_indices(y, self.missing_label)
        y[is_unlbld[:,0], is_unlbld[:,1]] = -1
        return NeuralNet.fit(self, X, y, **fit_params)

    def predict_annotator_perf(self, X, return_confusion_matrix=False):
        n_annotators = self.module__n_annotators
        P_class, logits_annot = self.forward(X)
        P_class = P_class.numpy()
        P_annot = F.softmax(logits_annot, dim=1)
        P_annot = P_annot.numpy()
        P_perf = np.array([np.einsum("ij,ik->ijk", P_class, P_annot[:, :, i]) for i in range(n_annotators)])
        P_perf = P_perf.swapaxes(0, 1)
        if return_confusion_matrix:
            return P_perf
        return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

    def predict(self, X):
        # maybe flag to switch between mode
        p_class, logits_annot = self.forward(X)
        return p_class.argmax(axis=1)

    def predict_proba(self, X):
        P_class, logits_annot = self.forward(X)
        P_class = P_class.numpy()
        return P_class

    def validation_step(self, batch, **fit_params):
        # not for loss but for acc
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.predict(Xi)
            acc = torch.mean(y_pred == yi)
        return {
            'loss': acc,
            'y_pred': y_pred,
        }


class CrowdLayerModule(nn.Module):
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
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_logits_annot: bool, optional (default=True)
            Flag whether the annotation logits are to be returned, next to the class-membership probabilities.

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