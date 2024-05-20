import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def predict(p_class):
    y_pred = p_class.argmax(axis=1).numpy()
    return y_pred

def perdict_annotation(logits_annot):
    P_annot = F.softmax(logits_annot, dim=1)
    P_annot = P_annot.detach().numpy()
    return P_annot

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

def plot_train_loss(path:str):
    metrics = pd.read_csv(path)
    metrics = metrics.groupby('epoch')['train_loss_epoch'].agg('mean')
    metrics.plot()
    plt.show()

if __name__ == '__main__':
    plot_train_loss('/Users/chengjiaying/BachelorProjekt/Crowd_Layer/lightning_logs/version_12/metrics.csv')
