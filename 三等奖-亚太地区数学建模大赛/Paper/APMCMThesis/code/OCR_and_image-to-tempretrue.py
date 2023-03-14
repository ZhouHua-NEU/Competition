!pip install paddlepaddle==2.3.2
!pip install "paddleocr>=2.0.1"

from paddleocr import PaddleOCR, draw_ocr
from glob import glob
import pandas as pandas_my
from tqdm import tqdm
from PIL import Image
import zipfile
import os
import cv2
import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from matplotlib import pyplot as plt
from math import ceil
from torchvision import transforms as tr
import torch.nn as nn
from torchvision import transforms, datasets
import json
import torch.optim as optim
import time
from torch.optim import lr_scheduler
import torchvision.models as models


# 图片分割
class MyData(Dataset):
    def __init__(self, filepath, transform=None):
        self.transform = transform
        self.root_path = filepath
        self.file_path = os.listdir(self.root_path)

    def __getitem__(self, index) -> T_co:
        temp = Image.open(os.path.join(self.root_path, self.file_path[index])).convert("RGB")
        x = np.asarray(temp.crop((1, 1, 102, 56))).transpose(2, 0, 1)
        x = torch.tensor(x / 255)
        y = self.file_path[index]
        return x, y

    def __len__(self):
        return len(self.file_path)


mydata = MyData("../Attachment1")
data = DataLoader(mydata, batch_size=10, drop_last=False)

for i in data:
    x, y = i
    for j in range(len(i[0])):
        print(j)
        torchvision.utils.save_image(x[j], os.path.join("..\\label\\", str(y[j])))


# 文字识别并保存
ocr = PaddleOCR(use_angle_cls=False, lang="ch")
image_path = glob('/kaggle/input/apcmc/label/label/*')
data=pandas_my.read_excel('/kaggle/input/apcmc/Attachment 2.xlsx')

!mkdir /kaggle/working/result
for i in tqdm(range(len(image_path))):
    path = image_path[i]
    result = ocr.ocr(path)
    
    data1=int(result[0][0][1][0][1:-1])
    data2=result[0][1][1][0][3:9]
    data3=result[0][2][1][0][3:9]
    
    if(not data2[-1].isdigit()):
        data2=data2[:-1]
        if(not data2[-1].isdigit()):data2=int(data2[:-1])
    if(not data2[0].isdigit()):
        data2=data2[1:]
        if(not data2[0].isdigit()):data2=int(data2[1:]) 
    data2=int(data2)

        
    if(not data3[-1].isdigit()):
        data3=data3[:-1]
        if(not data3[-1].isdigit()):data3=int(data3[:-1])
    if(not data3[0].isdigit()):
        data3=data3[1:]
        if(not data3[0].isdigit()):data3=int(data3[1:])
    data3=int(data3)
              
    data.iloc[data1-110,1:4]= pandas_my.DataFrame([data1,data2,data3],index=data.columns[1:4]).T
    
    result = result[0]
    image = Image.open(path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/kaggle/input/stkaiti/STKAITI.TTF')
    cv2.imwrite('/kaggle/working/result/%s.jpg'%data1,im_show)
    
data.astype(int).to_excel('./Attachment 2.xlsx',index=False)

dir_name = '/kaggle/working/result'
zip_file = dir_name+'.zip'
                                         
zip = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
for item in os.listdir(dir_name):
    zip.write(dir_name+os.sep+item)
zip.close()

# 预测温度一

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
        if(pred is not None):plt.title("time:{}, true:{}, predict:{}".format(idx.item(),label[i].item(),round(pred[i].item(), 3)))
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
        return img[:,15:,:],torch.tensor(self.label.iloc[int(self.x[index][-8:-4])-110,[2]])

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
optimizer = optim.Adam(net.parameters(), lr=0.000001,weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma = 0.9)


data=[]
for epoch in range(100): 
    #break
    train_loss = 0.0
    net.train()
    for step1,(x,label) in enumerate(train_loader):  
        optimizer.zero_grad()
        outputs = net(x.to(torch.float32).to(device))
        loss = loss_function(outputs, label.to(torch.float32).to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    data.append(train_loss)
      
    scheduler.step()
    print('epoch:',epoch+1,'train_loss:',train_loss)            
    torch.save(net, './Net1_%s.pth'%train_loss)

    if (train_loss<= 1 ):break

plot_curve(data[1:])


# 查看预测效果
#加载本地数据的类
class LocalDataset(Dataset):
    def __init__(self, data_list, label,transform=None):
        self.transform = transform
        self.x = data_list
        self.label = label

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('RGB')
        if self.transform is not None:img = self.transform(img)
        return img[:,15:,:],torch.tensor(self.label.iloc[int(self.x[index][-8:-4])-110,[2]]),int(self.x[index][-8:-4])

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

for step1,(x,y,idx) in enumerate(train_loader):
    if idx.item() == 116 or idx.item() == 126 or idx.item() == 136 or idx.item() == 666:
        plot_image(x, y,idx,net(x.to(device)))