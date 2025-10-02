#LIBRARIES
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer

#SCRIPTS
from Helpers import trainStep,testStep,printTrainTime,modelSummary,accuracy

from PretrainedModel import device, getModel

from Datas import trainDataLoader, testDataLoader

LEARNING_RATE = 0.0005

modelV1 = getModel(numClasses=10,
                   device=device).to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=modelV1.parameters(),
                            lr=LEARNING_RATE)

for param in modelV1.parameters(): 
    param.requires_grad = True 


torch.manual_seed(30)
epochs = 20

startTrainTimer = default_timer()

for epoch in tqdm(range(epochs)):
    trainStep(model=modelV1,
              dataLoader=trainDataLoader,
              optimizer=optimizer,
              lossFn=lossFunction,
              accFn=accuracy,
              device=device)
    
    testStep(model=modelV1,
             dataLoader=testDataLoader,
             lossFn=lossFunction,
             accFn=accuracy,
             device=device)
    
endTrainTimer = default_timer()

printTrainTime(startTimer=startTrainTimer,
               endTimer=endTrainTimer,
               device=device)

modelSum = modelSummary(model=modelV1,
                        dataLoader=testDataLoader,
                        lossFn=lossFunction,
                        accFn=accuracy,
                        device=device)

print(modelSum)

torch.save(modelV1.state_dict(),"pretrainedCIFAR10MODEL.pth")
print("Model ağırlıkları kaydedildi!")