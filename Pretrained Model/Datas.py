#LIBRARIES
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 64

transform =  transforms.Compose([
    transforms.Resize((224,224)), #RESNET 224X224 lük data bekliyor bizden dönüşüm yapıyoruz.
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainDatas = datasets.CIFAR10(
    root="datas",
    train=True,
    transform=transform,
    download=True
)

testDatas = datasets.CIFAR10(
    root="datas",
    train=False,
    transform=transform,
    download=True
)

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

