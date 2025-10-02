#LIBRARIES
import torch
from torch import nn
from torchvision import models

def getModel(numClasses = 3, device="cpu"):
    pretrainedModel = models.resnet18(pretrained = True)
    
    numFeatures = pretrainedModel.fc.in_features

    pretrainedModel.fc = nn.Linear(numFeatures,numClasses)

    pretrainedModel.to(device)
    return pretrainedModel

device = "cuda" if torch.cuda.is_available() else "cpu"