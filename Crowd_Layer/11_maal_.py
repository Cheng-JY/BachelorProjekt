import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from module.skorch_classifier import SkorchClassifier
from module.ground_truth_module import ClassifierModule

from skactiveml.pool import RandomSampling
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper

from skactiveml.utils import majority_vote, is_labeled, is_unlabeled
from skorch.callbacks import LRScheduler
from data_set.dataset import load_dataset_label_me, load_dataset_music

import torch
from torch import nn


if __name__ == '__main__':
    MISSING_LABEL = -1
    RANDOM_STATE = 0
    random_state = np.random.RandomState(RANDOM_STATE)

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    # dataset_name = 'music'
    # X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_music()
    dataset_name = 'label-me'
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_label_me()

    classes = np.unique(y_train_true)
    n_classes = len(classes)
    X_train = X_train.astype(np.float32)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]

    # performance
    accuracies = []

    # neural network
    hyper_parameter = {
        'max_epochs': 50,
        'batch_size': 64,
        'lr': 0.01,
        'optimizer__weight_decay': 0.0001,
    }
    lr_scheduler = LRScheduler(policy='CosineAnnealingLR', T_max=hyper_parameter['max_epochs'])

    # Randomly add missing labels
    y_partial = np.full_like(y_train, fill_value=MISSING_LABEL)
    initial_label_size = 100 # label me
    # initial_label_size = 32 # music

    for a_idx in range(n_annotators):
        is_lbld_a = is_labeled(y_train[:, a_idx], missing_label=-1)
        p_a = is_lbld_a / is_lbld_a.sum()  # no annotation with 0, with annotation 1/n_annotation
        initial_label_size_a = initial_label_size if is_lbld_a.sum() >= initial_label_size else is_lbld_a.sum()
        selected_idx_a = random_state.choice(np.arange(n_samples), size=initial_label_size_a, p=p_a, replace=False)
        y_partial[selected_idx_a, a_idx] = y_train[selected_idx_a, a_idx]

    # active learning
    sa_qs = RandomSampling(random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    ma_qs = SingleAnnotatorWrapper(sa_qs, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    candidate_indices = np.arange(n_samples)

    idx = lambda A: (A[:, 0], A[:, 1])

    n_cycle = 20
    # al_batch_size = 32  # music
    al_batch_size = 256  # label me

    A_random = np.ones_like(y_partial)
    annot_perf_sel = False

    for c in range(n_cycle+1):
        if c > 0:
            A_perf = A_random
            y_query = np.copy(y_partial)
            no_label_available = is_unlabeled(y_train, missing_label=MISSING_LABEL)
            # y_query[no_label_available] = 0 #Don't understand ???
            is_ulbld_query = is_unlabeled(y_query, missing_label=MISSING_LABEL)
            is_candidate = is_ulbld_query.all(axis=-1)
            candidates = candidate_indices[is_candidate]
            query_indices = ma_qs.query(
                X=X_train,
                y=y_query,
                candidates=candidate_indices,
                A_perf=A_perf,
                batch_size=al_batch_size,
                n_annotators_per_sample=1,
            )
            y_partial[idx(query_indices)] = y_train[idx(query_indices)]

        y_agg = majority_vote(y=y_partial, classes=classes, missing_label=MISSING_LABEL, random_state=RANDOM_STATE)

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

        net_mv.fit(X=X_train, y=y_agg)
        score = net_mv.score(X_test, y_test_true)
        accuracies.append(score)
        print('cycle ', c, score)

    plt.plot(accuracies)
    plt.title(f'{dataset_name}+majority-voting+random sampling')
    plt.show()
