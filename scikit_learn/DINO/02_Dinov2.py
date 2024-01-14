import torch
import torchvision
import torchvision.datasets as datasets

if __name__ == "__main__":

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
