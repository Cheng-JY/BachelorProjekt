import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == "__main__":
    # 1. Load Data Set
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize(224),
         transforms.CenterCrop(224)]
    )

    batch_size = 1

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

    all_embeddings = {}

    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            image, label = data

            embeddings = dinov2_vits14(image)
            all_embeddings[label[0]] = embeddings
            if i > 10:
                if i == 11:
                    test_embedding = dinov2_vits14(image)
                    test_label = label[0]
                break

    # 3. Train Linear Regression Model

    clf = LinearRegression()
    y = list(all_embeddings.keys())
    embedding_list = list(all_embeddings.values())
    clf.fit(np.array(embedding_list).reshape(-1, 384), y)

    # 4. Prediction
    prediction = clf.predict(np.array(test_embedding[0]).reshape(1,-1))

    print("predicted class: ", prediction[0])
    print("true class: ", test_label)






