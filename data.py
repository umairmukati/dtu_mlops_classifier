import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)

    train = np.load('./../../../data/corruptmnist/train_0.npz')
    test = np.load('./../../../data/corruptmnist/test.npz')
    return train, test
