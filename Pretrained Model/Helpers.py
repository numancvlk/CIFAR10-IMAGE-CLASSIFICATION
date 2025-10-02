#LIBRARIES
import torch

#SCRIPTS
from PretrainedModel import device

def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = correct / len(yTrue)
    return acc

def printTrainTime(startTimer,endTimer,device):
    totalTrainTime = endTimer - startTimer
    print(f"Train time is {totalTrainTime} on the {device}")

def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device):
    
    trainLoss, trainAccuracy = 0,0 
    model.train()
    for batch, (xTrain,yTrain) in enumerate(dataLoader):
        xTrain,yTrain = xTrain.to(device), yTrain.to(device)
        
        # FORWARD
        trainPred = model(xTrain)

        #LOSS / ACC
        loss = lossFn(trainPred,yTrain)
        trainLoss += loss.item()
        trainAccuracy += accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))

        #ZERO GRAD
        optimizer.zero_grad()

        #BACKWARD
        loss.backward()

        #STEP
        optimizer.step()

        if batch % 400 == 0:
            print(f"{len(xTrain) * batch} / {len(dataLoader.dataset)}")
    
    trainLoss /= len(dataLoader)
    trainAccuracy /= len(dataLoader)

    print(f"TRAIN LOSS = {trainLoss:.5f} | TRAIN ACCURACY = {trainAccuracy: .5f}%")

def testStep(model:torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accFn,
             device: torch.device = device):
    
    testLoss,testAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest, yTest = xTest.to(device), yTest.to(device)

            #FORWARD
            testPred = model(xTest)

            #LOSS / ACC
            loss = lossFn(testPred, yTest)
            testLoss += loss.item()

            testAccuracy += accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))

    
    testLoss /= len(dataLoader)
    testAccuracy /= len(dataLoader)
    print(f"TEST LOSS = {testLoss:.5f} | TEST ACCURACY = {testAccuracy:.5f}%")

def modelSummary(model:torch.nn.Module,
                 dataLoader: torch.utils.data.DataLoader,
                 lossFn: torch.nn.Module,
                 accFn,
                 device: torch.device = device):
    
    summaryLoss, summaryAccuracy = 0,0
    model.eval()

    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest, yTest = xTest.to(device), yTest.to(device)

            summaryPred = model(xTest)

            loss = lossFn(summaryPred,yTest)
            summaryLoss += loss.item()
            
            summaryAccuracy += accFn(yTrue = yTest, yPred = summaryPred.argmax(dim=1))

    summaryLoss /= len(dataLoader)
    summaryAccuracy /= len(dataLoader)
    return {"MODEL NAME": model.__class__.__name__,
            "MODEL LOSS": summaryLoss,
            "MODEL ACCURACY": summaryAccuracy}


