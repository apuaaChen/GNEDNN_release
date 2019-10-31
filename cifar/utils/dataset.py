import torch
import torchvision
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader

"""Random Dataset"""


class Dummy_dataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class DummyLoader:
    def __init__(self, size, length, batch_size):
        self.random_loader = DataLoader(dataset=Dummy_dataset(size, length),
                                        batch_size=batch_size, shuffle=True)


"""CIFAR-10"""


class Cifar10Provider:
    def __init__(self, root, batch=(128, 100), workers=4):
        self.root = root
        self.bs = batch
        self.workers = workers

        transform_train = transform.Compose([
            transform.RandomCrop(32, padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.root,
                                                train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.bs[0], shuffle=True,
                                                       num_workers=self.workers)

        testset = torchvision.datasets.CIFAR10(root=self.root,
                                               train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.bs[1], shuffle=False,
                                                      num_workers=self.workers)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
