import torch.nn.functional as F
import numpy as np
import torch

def predict(p_class):
    y_pred = p_class.argmax(axis=1).numpy()
    return y_pred

def predict_annotator_perf(P_class, logits_annot, return_confusion_matrix=False):
    n_annotators = logits_annot.shape[2]
    P_class = P_class.detach().numpy()
    P_annot = F.softmax(logits_annot, dim=1)
    P_annot = P_annot.detach().numpy()
    P_perf = np.array([np.einsum("ij,ik->ijk", P_class, P_annot[:, :, i]) for i in range(n_annotators)])
    P_perf = P_perf.swapaxes(0, 1)
    if return_confusion_matrix:
        return P_perf
    return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
