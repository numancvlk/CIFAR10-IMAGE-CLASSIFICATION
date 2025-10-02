#SCRIPTS
from MyModel import device, MYCIFAR10MODEL
from Datas import testDatas

#LIBRARIES
import torch
import random
import matplotlib.pyplot as plt

newModel = MYCIFAR10MODEL(inputShape=3,
                       hiddenUnit=128,
                       outputShape=len(testDatas.classes))

newModel.load_state_dict(torch.load("My_Model\myCIFAR10MODEL.pth"))

def makePredictions(model:torch.nn.Module,
                    data:list,
                    device:torch.device=device):

  predProbs = []
  model.to(device)
  model.eval()

  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device)

      predLogits = model(sample)

      predProb = torch.softmax(predLogits.squeeze(), dim=0)

      predProbs.append(predProb.cpu())

  return torch.stack(predProbs)



random.seed(41)
testSamples = []
testLabels = []

for sample, label in random.sample(list(testDatas), k=9):
  testSamples.append(sample)
  testLabels.append(label)


prediction = makePredictions(model=newModel,
                             data=testSamples,
                             device=device)

predictionClasses = prediction.argmax(dim=1)


# PLOT PREDICTIONS WITH COLORS
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(nrows, ncols, i+1)

    # Görüntüyü göster
    plt.imshow(sample.permute(1,2,0), cmap="gray")
    plt.axis('off')  # Eksenleri gizle

    # Tahmin ve gerçek sınıf isimleri
    predLabel = testDatas.classes[predictionClasses[i]]  # modelin tahmini
    trueLabel = testDatas.classes[testLabels[i]]         # gerçek sınıf

    # Renk belirle
    color = "green" if predLabel == trueLabel else "red"

    # Başlık ekle
    plt.title(f"P: {predLabel}\nT: {trueLabel}", color=color, fontsize=10)

plt.tight_layout()
plt.show()