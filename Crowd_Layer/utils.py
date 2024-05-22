import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

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

def get_experiment_result(uri, experiment_name, path):
    mlflow.set_tracking_uri(uri=uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    csv_df = df[['params.model_name', 'params.batch_size', 'params.max_epochs', 'params.lr',
                 'params.optimizer__weight_decay', 'metrics.train_accuracy', 'metrics.test_accuracy', 'metrics.time']]
    csv_df.to_csv(path, index=False)

def write_experiment_result(tensor, path):
    numpy_array = tensor.detach().numpy()
    np.savetxt(path, numpy_array[0])


if __name__ == '__main__':
    # plot_train_loss('/Users/chengjiaying/BachelorProjekt/Crowd_Layer/lightning_logs/version_0/metrics.csv')
    uri_training = '/Users/chengjiaying/BachelorProjekt/Crowd_Layer/tracking'
    exp_training = 'Crowd-Layer-Training'
    path_training = 'training.csv'
    get_experiment_result(uri_training, exp_training, path_training)
