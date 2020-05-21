import torch
import os
from urllib import request
from scipy.io import loadmat
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_elevators_data():
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


def get_vision_data():
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    common_trans = [transforms.ToTensor(), normalize]
    train_compose = transforms.Compose(aug_trans + common_trans)
    test_compose = transforms.Compose(common_trans)


    dataset = "cifar10"

    if ('CI' in os.environ):  # this is for running the notebook in our testing framework
        train_set = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.rand(8).round().long())
        test_set = torch.utils.data.TensorDataset(torch.randn(4, 3, 32, 32), torch.rand(4).round().long())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)
        num_classes = 2
    elif dataset == 'cifar10':
        train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
        test_set = dset.CIFAR10('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        num_classes = 10
    elif dataset == 'cifar100':
        train_set = dset.CIFAR100('data', train=True, transform=train_compose, download=True)
        test_set = dset.CIFAR100('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        num_classes = 100
    else:
        raise RuntimeError('dataset must be one of "cifar100" or "cifar10"')

    return train_loader, test_loader, num_classes



def get_pyro_data():
    # Here we specify a 'true' latent function lambda
    scale = lambda x: np.sin(2 * np.pi * x) + 1

    # Generate synthetic data
    # here we generate some synthetic samples
    NSamp = 100

    X = np.linspace(0, 1, NSamp)

    #fig, (lambdaf, samples) = plt.subplots(1, 2, figsize=(10, 3))

    # lambdaf.plot(X, scale(X))
    # lambdaf.set_xlabel('x')
    # lambdaf.set_ylabel('$\lambda$')
    # lambdaf.set_title('Latent function')

    Y = np.zeros_like(X)
    for i, x in enumerate(X):
        Y[i] = np.random.exponential(scale(x), 1)

    return X, Y