import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset():
    data_dir = "./music-multi-annotator-dataset"

    X_train = np.load(f'{data_dir}/music-X.npy')
    y_train = np.load(f'{data_dir}/music-y.npy')
    y_train_true = np.load(f'{data_dir}/music-y-true.npy')
    X_valid = np.load(f'{data_dir}/music-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/music-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/music-X-test.npy')
    y_test_true = np.load(f'{data_dir}/music-y-true-test.npy')

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true


class MusicDataSet(Dataset):
    def __init__(self, type: str, n_annotators=44):
        super(MusicDataSet, self).__init__()

        self.type = type

        X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset()

        y_train_new = np.column_stack([y_train_true for _ in range(n_annotators)])
        y_train_new[:, -1] = 0


        if type == 'train':
            self.X = X_train
            self.y = y_train
        elif type == 'valid':
            self.X = X_valid
            self.y = y_valid_true
        elif type == 'test':
            self.X = X_test
            self.y = y_test_true
        elif type == 'fake':
            self.X = X_train
            self.y = y_train_new

        self.y_train_true = y_train_true

    def return_X_y(self):
        return torch.tensor(self.X).float(), torch.tensor(self.y).long()

    def return_X_y_numpy(self):
        return self.X, self.y

    def return_y_train_true(self):
        return torch.tensor(self.y_train_true).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        return torch.tensor(X).float(), torch.tensor(y).long()
