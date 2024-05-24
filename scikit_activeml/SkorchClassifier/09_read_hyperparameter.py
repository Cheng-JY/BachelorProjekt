import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import mlflow


def plot_loss(df, nn_name, max_epochs, batch_size, optimizer_lr, optimizer__weight_decay):
    artifact_uri = df.loc[(df['params.nn_name'] == nn_name)
                          & (df['params.batch_size'] == str(batch_size))
                          & (df['params.max_epochs'] == str(max_epochs))
                          & (df['params.optimizer_lr'] == str(optimizer_lr))
                          & (df['params.optimizer__weight_decay'] == str(optimizer__weight_decay))
                          ]
    uri = artifact_uri.iloc[0]['artifact_uri']
    artifact = os.path.join(uri, 'result.csv')

    if os.path.exists(artifact):
        result = pd.read_csv(artifact, index_col=False)
        train_loss = result.loc[:, 'train_loss']
        valid_loss = result.loc[:, 'valid_loss']
        plt.plot(train_loss, label='train_loss')
        plt.plot(valid_loss, label='valid_loss')
        plt.show()

def get_best_hyperparameters(df, nn_name):
    df = df.loc[(df['params.nn_name'] == nn_name)]
    print(df.columns)
    max_test_accuracy = df.loc[:, 'metrics.test_accuracy'].max()
    hyperparameter = df.loc[df['metrics.test_accuracy'] == max_test_accuracy, ]
    hyperparameter = hyperparameter[['params.batch_size', 'params.max_epochs',
                          'params.lr', 'params.optimizer__weight_decay',
                          'metrics.train_accuracy', 'metrics.test_accuracy']]
    hyperparameter.to_csv(f'{nn_name}_best_hyperparameter.csv', index=False)


if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="/Users/chengjiaying/BachelorProjekt/scikit_activeml/SkorchClassifier/tracking")

    experiment = mlflow.get_experiment_by_name("Hyperparameter-Tuning-with-lrs")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    plot_df = df[['params.nn_name', 'params.batch_size', 'params.max_epochs', 'params.lr',
                  'params.optimizer__weight_decay', 'metrics.train_accuracy', 'metrics.test_accuracy', 'artifact_uri']]
    csv_df = df[['params.nn_name', 'params.batch_size', 'params.max_epochs', 'params.lr',
                 'params.optimizer__weight_decay', 'metrics.train_accuracy', 'metrics.test_accuracy']]
    csv_df.sort_values(by=['params.nn_name', 'params.lr', 'params.optimizer__weight_decay', 'metrics.train_accuracy',
                           'metrics.train_accuracy'])
    csv_df.to_csv('hyperparameter-with-lrs.csv', index=False)
    # plot_loss(plot_df, nn_name='lb', max_epochs=100, batch_size=16, optimizer_lr=0.005, optimizer__weight_decay=0.001)
    # get_best_hyperparameters(csv_df, nn_name='lb')
