import os
import sys
import time

from matplotlib import pyplot as plt
from skorch.callbacks import LRScheduler

sys.path.append('..')

import numpy as np
import pandas as pd
import torch

from torch import nn
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from module.crowd_layer_skorch import CrowdLayerSkorch
from module.skorch_classifier import SkorchClassifier
from module.ground_truth_module import ClassifierModule
from data_set.dataset import LabelMeDataSet, load_dataset_label_me

from skactiveml.utils import majority_vote

import mlflow


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    seed = 4
    MISSING_LABEL = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_label_me()

    dataset_classes = np.unique(y_test_true)
    print(dataset_classes)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]

    valid_ds = Dataset(X_valid, y_valid_true)

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/BachelorProjekt/Crowd_Layer/tracking')
    exp = mlflow.get_experiment_by_name(name='Label-Me-Training')
    experiment_id = mlflow.create_experiment(name='Label-Me-Training') if exp is None else exp.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        hyper_dict = {
            'max_epochs': 50,
            'batch_size': 64,
            'lr': 0.01,
            'optimizer__weight_decay': 0.0001
        }
        lr_scheduler = LRScheduler(policy="CosineAnnealingLR", T_max=hyper_dict['max_epochs'])

        nn_name = 'cl'
        if nn_name == 'cl':
            gt_net = ClassifierModule(n_classes=n_classes, n_features=n_features, dropout=0.5)
            net = CrowdLayerSkorch(
                module__n_classes=n_classes,
                module__n_annotators=n_annotators,
                module__gt_net=gt_net,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=seed,
                train_split=predefined_split(valid_ds),
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
        elif nn_name == 'ub' or nn_name == 'lb':
            net = SkorchClassifier(
                ClassifierModule,
                module__n_classes=n_classes,
                module__n_features=n_features,
                module__dropout=0.5,
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
            if nn_name == 'ub':
                y_train = y_train_true
            elif nn_name == 'lb':
                y_train = majority_vote(y_train, classes=dataset_classes, missing_label=-1)

        hyper_dict['nn_name'] = nn_name
        hyper_dict['seed'] = seed
        mlflow.log_params(hyper_dict)

        start = time.time()
        net.fit(X_train, y_train)
        end = time.time()

        train_accuracy = net.score(X_train, y_train_true)

        p_pred = net.predict_proba(X_test)
        print(p_pred[0])
        test_accuracy = net.score(X_test, y_test_true)
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'time': end-start,
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        history = net.history
        train_loss = history[:, 'train_loss']
        valid_loss = history[:, 'valid_loss']

        plt.plot(train_loss)
        plt.title(f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
        plt.show()

        loss = {'train_loss': train_loss, 'valid_loss': valid_loss}
        df = pd.DataFrame.from_dict(data=loss)
        outpath = active_run.info.artifact_uri
        outpath = os.path.join(outpath, "result.csv")
        df.to_csv(outpath, index=False)




