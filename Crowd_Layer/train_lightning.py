import pytorch_lightning as L  # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data
from data_set.dataset import *
from module.crowd_layer_lightning import CrowdLayerLightning
from module.ground_truth_module import ClassifierModule
from utils import *
from sklearn.metrics import accuracy_score
import mlflow
import time

if __name__ == '__main__':
    seed_everything(42)

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/BachelorProjekt/Crowd_Layer/tracking')
    # Crowd-Layer-4true1adversary
    # Crowd-Layer-Training
    exp = mlflow.get_experiment_by_name(name='Crowd-Layer-Training')
    experiment_id = mlflow.create_experiment(name='Crowd-Layer-Training') if exp is None else exp.experiment_id
    print(experiment_id)

    with mlflow.start_run(experiment_id=experiment_id):
        hyper_dict = {
            'max_epochs': 250,
            'batch_size': 64,
            'lr': 0.005,
            'optimizer__weight_decay': 0.001
        }

        n_classes = 10
        dropout = 0.0
        n_annotators = 44
        # n_annotators = 5

        train_dataset = MusicDataSet('train', n_annotators)
        # train_dataset = MusicDataSet('fake', n_annotators)
        valid_dataset = MusicDataSet('valid', n_annotators)
        test_dataset = MusicDataSet('test', n_annotators)

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
                                       weight_decay=hyper_dict['optimizer__weight_decay'],
                                       max_epochs=hyper_dict['max_epochs'])

        hyper_dict['model_name'] = 'CrowdLayerLightning'
        mlflow.log_params(hyper_dict)

        # trainer = L.Trainer(max_epochs=hyper_dict['max_epochs'], barebones=True)
        # for compare with pytorch
        trainer = L.Trainer(max_epochs=hyper_dict['max_epochs'])
        start = time.time()
        trainer.fit(cl_model, train_dataloaders=train_dl)
        end = time.time()

        with torch.no_grad():
            cl_model.eval()
            p_class_train, logits_annot_train = cl_model(X_train)
            y_pred_train = predict(p_class_train)
            train_accuracy = accuracy_score(y_train_true.numpy(), y_pred_train)

            p_class_test, logits_annot_test = cl_model(X_test)
            y_pred_test = predict(p_class_test)
            test_accuracy = accuracy_score(y_test.numpy(), y_pred_test)
            P_anno = F.softmax(logits_annot_test, dim=1)

            # write_experiment_result(p_class_test, 'p_class_lightning.csv')
            # write_experiment_result(P_anno, 'p_annot_lightning.csv')

            metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'time': end - start
            }

            mlflow.log_metrics(metrics)
            print(metrics)

            print(cl_model.annotator_layers[0].weight)

            proba_annotator_pref = predict_annotator_perf(p_class_test[0:1], logits_annot_test[0:1])
            print(proba_annotator_pref)

