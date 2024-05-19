import sys

from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('..')

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from module.crowd_layer_pytorch import CrowdLayerPytorch
from module.ground_truth_module import ClassifierModule
from data_set.dataset import MusicDataSet
from utils import *

import mlflow

if __name__ == '__main__':
    seed_everything(42)

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/BachelorProjekt/Crowd_Layer/tracking')
    exp = mlflow.get_experiment_by_name(name='Crowd-Layer-05-19')
    experiment_id = mlflow.create_experiment(name='Crowd-Layer-05-19') if exp is None else exp.experiment_id
    print(experiment_id)

    with mlflow.start_run(experiment_id=experiment_id):

        hyper_dict = {
            'max_epochs': 250,
            'batch_size': 64,
            'lr': 0.001,
            'optimizer__weight_decay': 0.0001
        }

        n_classes = 10
        dropout = 0.0
        n_annotators = 44

        train_dataset = MusicDataSet('train')
        valid_dataset = MusicDataSet('valid')
        test_dataset = MusicDataSet('test')

        train_dataloader = DataLoader(train_dataset, batch_size=hyper_dict['batch_size'], drop_last=True, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=hyper_dict['batch_size'], drop_last=True, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=hyper_dict['batch_size'], drop_last=True, shuffle=False)

        X_train, y_train = train_dataset.return_X_y()
        y_train_true = train_dataset.return_y_train_true()
        X_valid, y_valid = valid_dataset.return_X_y()
        X_test, y_test = test_dataset.return_X_y()

        gt_model = ClassifierModule(n_classes=n_classes, dropout=dropout)
        cl_model = CrowdLayerPytorch(n_classes=n_classes, n_annotators=n_annotators, gt_net=gt_model)
        optimizer = optim.Adam(cl_model.parameters(), lr=hyper_dict['lr'], weight_decay=hyper_dict['optimizer__weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_dict['max_epochs'])

        hyper_dict['model_name'] = 'CrowdLayerPytorch'
        mlflow.log_params(hyper_dict)

        # Training Process
        loss_progess = []
        for i in range(hyper_dict['max_epochs']):
            cl_model.train()
            epoch_loss = 0
            for X_batch, y_batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                p_class, logits_annot = cl_model(X_batch)
                loss = nn.functional.cross_entropy(logits_annot, y_batch, reduction='mean', ignore_index=-1)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(train_dataloader)
            loss_progess.append(epoch_loss)

        with torch.no_grad():
            p_class_train, logits_annot_train = cl_model(X_train)
            y_pred_train = predict(p_class_train)
            train_accuracy = accuracy_score(y_train_true.numpy(), y_pred_train)

            p_class_test, logits_annot_test = cl_model(X_test)
            y_pred_test = predict(p_class_test)
            test_accuracy = accuracy_score(y_test.numpy(), y_pred_test)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        }

        mlflow.log_metrics(metrics)
        print(metrics)

        print(cl_model.annotator_layers[0].weight)

        proba_annotator_pref = predict_annotator_perf(p_class_test[0:1], logits_annot_test[0:1])
        print(proba_annotator_pref)

        plt.plot(loss_progess)
        plt.title(f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
        plt.show()

