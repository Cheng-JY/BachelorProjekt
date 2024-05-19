import torch  # torch will allow us to create tensors.

import pytorch_lightning as L  # lightning has tons of cool tools that make neural networks easier
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data
from data_set.dataset import *
from module.crowd_layer_lightning import CrowdLayerLightning
from module.ground_truth_module import ClassifierModule
from utils import *
from sklearn.metrics import accuracy_score
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
            'lr': 0.01,
            'optimizer__weight_decay': 0.0001
        }

        n_classes = 10
        dropout = 0.0
        n_annotators = 44

        train_dataset = MusicDataSet('train')
        valid_dataset = MusicDataSet('valid')
        test_dataset = MusicDataSet('test')

        X_train, y_train = train_dataset.return_X_y()
        y_train_true = train_dataset.return_y_train_true()
        X_valid, y_valid = valid_dataset.return_X_y()
        X_test, y_test = test_dataset.return_X_y()

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=hyper_dict['batch_size'], shuffle=True)
        valid_ds = TensorDataset(X_valid, y_valid)
        test_ds = TensorDataset(X_test, y_test)

        gt_model = ClassifierModule(n_classes=n_classes, dropout=dropout)
        cl_model = CrowdLayerLightning(n_classes=n_classes, n_annotators=n_annotators, gt_net=gt_model,
                                       lr=hyper_dict['lr'],
                                       weight_decay=hyper_dict['optimizer__weight_decay'])

        hyper_dict['model_name'] = 'CrowdLayerLightning'
        mlflow.log_params(hyper_dict)

        trainer = L.Trainer(max_epochs=hyper_dict['max_epochs'])
        trainer.fit(cl_model, train_dataloaders=train_dl)

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

        # Retrieve logged metrics
        print(trainer.logged_metrics)

        # Plotting the losses
        # print(train_loss)
        # plt.plot(train_loss)
        # plt.title(f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
        # plt.show()
