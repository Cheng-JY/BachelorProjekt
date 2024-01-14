import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == "__main__":
    # 1. Load Data Set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
