import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # 1. Load Data Set
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    batch_size = 4

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. Compute Embeddings for Image
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, label = data

            embeddings = dinov2_vits14(image)
            embeddings_list.append(embeddings)
            labels_list.append(label)
            if i > 100:
                break

    X = torch.cat(embeddings_list, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()
    print(X.shape)
    print(y.shape)

    # X = np.load('cifar10_dinov2_X.npy')
    # y = np.load('cifar10_dinov2_y.npy')

    # 3. Train Linear Regression Model
    n_train_samples = int(len(X) * 0.8)
    clf = LogisticRegression(max_iter=5000)
    # y = np.array(labels_list)
    # X = np.array(embeddings_list).reshape(-1, 384)
    score = clf.fit(X[:n_train_samples], y[:n_train_samples]).score(X[n_train_samples:], y[n_train_samples:])
    print(score)

    np.save('cifar10_dinov2_X.npy', X)
    np.save('cifar10_dinov2_y.npy', y)
