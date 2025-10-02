#LIBRARIES
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 32

#COLLECTING DATAS
trainDatas = datasets.CIFAR10(
    root="datas",
    train=True,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

testDatas = datasets.CIFAR10(
    root="datas",
    train=False,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

#DATALAODERS
trainDataLoader = DataLoader(
    dataset=trainDatas,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testDataLoader = DataLoader(
    dataset=testDatas,
    batch_size=BATCH_SIZE,
    shuffle=False
)
