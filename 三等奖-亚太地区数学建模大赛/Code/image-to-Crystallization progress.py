import torch
from matplotlib import pyplot as plt
from math import ceil
import os
from torchvision import transforms as tr
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from glob import glob
import pandas as pd
import torch.nn as nn
from torchvision import transforms, datasets
import json
import torch.optim as optim
import time
from torch.optim import lr_scheduler
import torchvision.models as models

#画损失函数图
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.savefig("/kaggle/working/loss.jpg")
    plt.show()


#画出每一个batchsize中的图片
def plot_image(img,label,idx,pred=None):
#     pred = torch.tensor(round(pred.item(), 3))
    fig = plt.figure()
    for i in range(len(img)):
        plt.subplot(ceil(len(img)**0.5), ceil(len(img)**0.5), i + 1)
        plt.tight_layout()
        plt.imshow(img[i].transpose(0,1).transpose(1,2))
        if(pred is not None):plt.title("time:{}, true:{}, predict:{}".format(idx.item(),round(label[i].item(),3),round(pred[i].item(), 3)))
        else:plt.title("{} D.C.".format(label[i].item()))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('/kaggle/working/{}.jpg'.format(idx.item()))
    plt.show()

#加载本地数据的类
class LocalDataset(Dataset):
    def __init__(self, data_list, label,transform=None):
        self.transform = transform
        self.x = data_list
        self.label = label

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('RGB')
        if self.transform is not None:img = self.transform(img)
        return img[:,15:,:],torch.tensor(self.label.iloc[int(self.x[index][-8:-4])-110,[5]]), int(self.x[index][-8:-4])

    def __len__(self):
        return len(self.x)

#加载本地的数据集
def local_datasets(data_path,label,bt,PM,NM=0):
    numclass_path = glob(os.path.join(data_path, '*'))

    data_transform = {"train": tr.Compose([
        tr.Resize((329,428)),
        tr.ToTensor()])}

    train_dataset = LocalDataset(numclass_path,label,transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bt, shuffle=False,num_workers=NM,pin_memory=PM)

    return train_loader

label = pd.read_excel("/kaggle/input/apcmc/Attachment 2.xlsx")
train_loader= local_datasets('/kaggle/input/apcmc/Attachment 1/Attachment 1',label,1,True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

resnet18_regress = models.resnet18(pretrained=False)
resnet18_regress.fc=torch.nn.Linear(in_features=512, out_features=1, bias=True)

net=resnet18_regress.to(device)
# net = torch.load("/kaggle/working/Net1_961.6232820264995.pth",map_location=device)

less_valloss = 10000.0
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma = 0.9)

data=[]
for epoch in range(100): 
    #break
    train_loss = 0.0
    net.train()
    for step1,(x,label,index)in enumerate(train_loader):
        if index >= 150:
            optimizer.zero_grad()
            outputs = net(x.to(torch.float32).to(device))
            loss = loss_function(outputs, label.to(torch.float32).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    data.append(train_loss)

    scheduler.step()
    print('epoch:',epoch+1,'train_loss:',train_loss)
    torch.save(net, './Net2_%s.pth'%train_loss)

plot_curve(data[1:])

for step1,(x,y,idx) in enumerate(train_loader):
    if idx.item() == 366 or idx.item() == 266 or idx.item() == 466 or idx.item() == 666:
        plot_image(x, y,idx,net(x.to(device)))