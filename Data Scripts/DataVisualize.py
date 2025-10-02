#LIBRARIES
import torch
import matplotlib.pyplot as plt

#SCRIPTS
from Datas import trainDatas

#DATA VISUALIZE
torch.manual_seed(30)

fig = plt.figure(figsize=(10,10))
rows,cols = 5,5

for i in range(1, rows*cols+1):
    randomIndex = torch.randint(0,len(trainDatas),size=[1]).item()
    image, label = trainDatas[randomIndex]
    ax = fig.add_subplot(rows,cols,i)
    ax.set_title(trainDatas.classes[label])
    ax.imshow(image.permute(1,2,0)) 
    ax.axis("off")

plt.show()