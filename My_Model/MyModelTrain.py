#LIBRARIES
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer

#SCRIPTS
from Helpers import trainStep,testStep,printTrainTime,modelSummary,accuracy

from MyModel import MYCIFAR10MODEL, device

from Datas import trainDatas, trainDataLoader, testDataLoader

LEARNING_RATE = 0.01

modelV0 = MYCIFAR10MODEL(inputShape=3,
                         hiddenUnit=128,
                         outputShape=len(trainDatas.classes)).to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelV0.parameters(),
                            lr=LEARNING_RATE)


torch.manual_seed(30)
epochs = 40

startTrainTimer = default_timer()

for epoch in tqdm(range(epochs)):
    trainStep(model=modelV0,
              dataLoader=trainDataLoader,
              optimizer=optimizer,
              lossFn=lossFunction,
              accFn=accuracy,
              device=device)
    
    testStep(model=modelV0,
             dataLoader=testDataLoader,
             lossFn=lossFunction,
             accFn=accuracy,
             device=device)
    
endTrainTimer = default_timer()

printTrainTime(startTimer=startTrainTimer,
               endTimer=endTrainTimer,
               device=device)

modelSum = modelSummary(model=modelV0,
                        dataLoader=testDataLoader,
                        lossFn=lossFunction,
                        accFn=accuracy,
                        device=device)

print(modelSum)

torch.save(modelV0.state_dict(),"myCIFAR10MODEL.pth")
print("Model ağırlıkları kaydedildi!")