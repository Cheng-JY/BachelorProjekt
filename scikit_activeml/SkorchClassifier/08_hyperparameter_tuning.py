import os
import sys

from skorch.callbacks import LRScheduler

sys.path.append('..')

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from classifier.crowd_layer_classifier import CrowdLayerClassifier
from classifier.skorch_classifier import SkorchClassifier

from skactiveml.utils import majority_vote

import mlflow


class GroundTruthModule(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(GroundTruthModule, self).__init__()
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


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets():
    data_dir = "./music-multi-annotator-dataset"

    X_train = np.load(f'{data_dir}/music-X.npy')
    y_train = np.load(f'{data_dir}/music-y.npy')
    y_train_true = np.load(f'{data_dir}/music-y-true.npy')
    X_valid = np.load(f'{data_dir}/music-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/music-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/music-X-test.npy')
    y_test_true = np.load(f'{data_dir}/music-y-true-test.npy')

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true



if __name__ == '__main__':
    seed = 1
    MISSING_LABEL = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_datasets()

    dataset_classes = np.unique(y_test_true)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]

    valid_ds = Dataset(X_valid, y_valid_true)

    mlflow.set_tracking_uri(uri="/Users/chengjiaying/BachelorProjekt/scikit_activeml/SkorchClassifier/tracking")
    exp = mlflow.get_experiment_by_name(name="Hyperparameter-Tuning-with-lrs")
    experiment_id = mlflow.create_experiment(name="Hyperparameter-Tuning-with-lrs") if exp is None else exp.experiment_id

    with (mlflow.start_run(experiment_id=experiment_id) as active_run):
        hyper_dict = {
            'max_epochs': 50,
            'batch_size': 64,
            'lr': 0.01,
            'optimizer__weight_decay': 0.0
        }
        lr_scheduler = LRScheduler(policy="CosineAnnealingLR", T_max=hyper_dict['max_epochs'])

        nn_name = 'ub'
        if nn_name == 'cl':
            gt_net = GroundTruthModule(n_classes=n_classes, dropout=0.5)
            net = CrowdLayerClassifier(
                module__n_annotators=y_train.shape[1],
                module__gt_net=gt_net,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=seed,
                train_split=predefined_split(valid_ds),
                verbose=False,
                optimizer=torch.optim.AdamW,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
        elif nn_name == 'ub':
            net = SkorchClassifier(
                GroundTruthModule,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=1,
                criterion=nn.CrossEntropyLoss(),
                train_split=predefined_split(valid_ds),
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
            y_train = y_train_true
        elif nn_name == 'lb':
            net = SkorchClassifier(
                GroundTruthModule,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=1,
                criterion=nn.CrossEntropyLoss(),
                train_split=predefined_split(valid_ds),
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
            y_train = majority_vote(y_train, classes=dataset_classes, missing_label=-1)
        net.initialize()
        print(net.lr)

        hyper_dict['nn_name'] = nn_name
        mlflow.log_params(hyper_dict)

        net.fit(X_train, y_train)

        y_train_pred = net.predict(X_train)
        train_accuracy = accuracy_score(y_train_true, y_train_pred)

        y_pred = net.predict(X_test)
        test_accuracy = accuracy_score(y_pred, y_test_true)
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        history = net.history
        train_loss = history[:, 'train_loss']
        valid_loss = history[:, 'valid_loss']

        loss = {'train_loss': train_loss, 'valid_loss': valid_loss}
        df = pd.DataFrame.from_dict(data=loss)
        outpath = active_run.info.artifact_uri
        outpath = os.path.join(outpath, "result.csv")
        df.to_csv(outpath, index=False)
