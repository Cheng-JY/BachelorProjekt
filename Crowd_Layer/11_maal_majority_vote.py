import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

from module.crowd_layer_skorch import CrowdLayerSkorch
from module.skorch_classifier import SkorchClassifier
from module.ground_truth_module import ClassifierModule
from skactiveml.pool import RandomSampling
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.utils import majority_vote
from skorch.callbacks import LRScheduler
from data_set.dataset import LabelMeDataSet, load_dataset_label_me

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == '__main__':
    MISSING_LABEL = -1
    RANDOM_STATE = 0

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_label_me()

    classes = np.unique(y_train_true)
    n_classes = len(classes)
    X_train = X_train.astype(np.float32)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]

    # neural network
    hyper_parameter = {
        'max_epochs': 50,
        'batch_size': 64,
        'lr': 0.01,
        'optimizer__weight_decay': 0.0001,
    }
    lr_scheduler = LRScheduler(policy='CosineAnnealingLR', T_max=hyper_parameter['max_epochs'])

    net_mv = SkorchClassifier(
        ClassifierModule,
        module__n_classes=n_classes,
        module__n_features=n_features,
        module__dropout=0.5,
        classes=classes,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=1,
        criterion=nn.CrossEntropyLoss(),
        train_split=None,
        verbose=False,
        optimizer=torch.optim.RAdam,
        device=device,
        callbacks=[lr_scheduler],
        **hyper_parameter
    )

    # active learning
    sa_qs = RandomSampling(random_state=0, missing_label=MISSING_LABEL)
    ma_qs = SingleAnnotatorWrapper(sa_qs, random_state=0, missing_label=MISSING_LABEL)

    idx = lambda A: (A[:, 0], A[:, 1])

    n_cycle = 20

    # the already observed labels for each sample and annotator
    y = np.full(shape=(n_samples, n_annotators), fill_value=MISSING_LABEL, dtype=np.int32)
    y_init = np.full_like(y_train_true, fill_value=MISSING_LABEL, dtype=np.int32)

    query_idx = sa_qs.query(X_train, y_init, batch_size=256)
    y[query_idx] = y_train[query_idx]

    accuracies = []
    y_mv = majority_vote(y, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    net_mv.fit(X_train, y_mv)
    score = net_mv.score(X_test, y_test_true)
    accuracies.append(score)
    print(score)

    for c in range(n_cycle):
        query_idx = ma_qs.query(X_train, y, batch_size=512, n_annotators_per_sample=2)
        y[idx(query_idx)] = y_train[idx(query_idx)]
        y_mv = majority_vote(y, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
        net_mv.fit(X_train, y_mv)
        score = net_mv.score(X_test, y_test_true)
        accuracies.append(score)
        print('cycle ', c, score)

    plt.plot(accuracies)
    plt.show()
