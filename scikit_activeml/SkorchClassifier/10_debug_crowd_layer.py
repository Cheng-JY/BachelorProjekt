import os
import sys

from matplotlib import pyplot as plt
from skorch.callbacks import LRScheduler

sys.path.append('..')

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score
from torch import nn
import torch.nn.functional as F
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from classifier.crowd_layer_classifier import CrowdLayerClassifier
from classifier.skorch_classifier import SkorchClassifier

from skactiveml.utils import majority_vote

import mlflow


class GroundTruthModule(nn.Module):
    def __init__(self, n_classes, dropout):
        super(GroundTruthModule, self).__init__()
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

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true


if __name__ == '__main__':
    seed = 42
    MISSING_LABEL = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_datasets()

    dataset_classes = np.unique(y_test_true)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]

    valid_ds = Dataset(X_valid, y_valid_true)

    hyper_dict = {
        'max_epochs': 250,
        'batch_size': 64,
        'lr': 0.01,
        'optimizer__weight_decay': 0.0001
    }

    lr_scheduler = LRScheduler(policy="CosineAnnealingLR", T_max=hyper_dict['max_epochs'])
    gt_net = GroundTruthModule(n_classes=n_classes, dropout=0.0)
    net = CrowdLayerClassifier(
        module__n_annotators=44,
        module__gt_net=gt_net,
        classes=dataset_classes,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=1,
        train_split=predefined_split(valid_ds),
        verbose=False,
        optimizer=torch.optim.AdamW,
        device=device,
        callbacks=[lr_scheduler],
        **hyper_dict,
    )
    net.initialize()
    print(net.lr)

    y_train_new = np.column_stack([y_train_true for _ in range(5)])
    net.fit(X_train, y_train)

    y_train_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train_true, y_train_pred)

    y_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_pred, y_test_true)
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
    }

    print(metrics)

    # print(net.module_.annotator_layers[0].weight)

    # proba_annotator_pref = net.predict_annotator_perf(X_test[0:1], False)
    # print(proba_annotator_pref)

    history = net.history
    train_loss = history[:, 'train_loss']
    plt.plot(train_loss)
    plt.title(f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
    plt.show()