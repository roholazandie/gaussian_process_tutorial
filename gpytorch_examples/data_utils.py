import torch
import os
from urllib import request
from scipy.io import loadmat
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def get_data():
    if not os.path.isfile('../elevators.mat'):
        print('Downloading \'elevators\' UCI dataset...')
        request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk',
                                   '../elevators.mat')

    data = torch.Tensor(loadmat('../../datasets/elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    n_train = int(np.floor(0.8 * len(X)))
    x_train = X[:n_train, :].contiguous()
    y_train = y[:n_train].contiguous()

    x_test = X[n_train:, :].contiguous()
    y_test = y[n_train:].contiguous()

    if torch.cuda.is_available():
        x_train, y_train, x_test, y_test = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda()

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    return train_loader, test_loader, x_train, x_test, y_test