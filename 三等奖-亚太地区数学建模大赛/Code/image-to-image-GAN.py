import torch
from matplotlib import pyplot as plt
from math import ceil
import os
from glob import glob
import torch
from torchvision import transforms as tr
from torch.utils.data import DataLoader,Dataset
from PIL import Image


#加载本地数据的类
class LocalDataset(Dataset):
    def __init__(self, data_list, label,transform=None):
        self.transform = transform
        self.x = data_list
        self.label = label

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('RGB')
        if self.transform is not None:img = self.transform(img)
        return img[:,15:,:],(int(self.x[index][-8:-4])-110)

    def __len__(self):
        return len(self.x)



#加载本地的数据集
def local_datasets(data_path,label,bt,PM,NM=0):
    numclass_path = glob(os.path.join(data_path, '*'))

    data_transform = {"train": tr.Compose([
        tr.Resize((329,329)),
        tr.ToTensor()])}

    train_dataset = LocalDataset(numclass_path,label,transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bt, shuffle=False,num_workers=NM,pin_memory=PM)

    return train_loader

import pandas as pd
label = pd.read_excel("/kaggle/input/apcmc/Attachment 2.xlsx")
train_loader= local_datasets('/kaggle/input/apcmc/Attachment 1/Attachment 1',label,1,True)

from torchvision.utils import save_image
!mkdir -p /kaggle/working/facadescd/train/
!mkdir -p /kaggle/working/facadescd/test/

for (img,index) in train_loader:
    if(index==len(train_loader)):continue
    save_image(img,"/kaggle/working/facadescd/train/%d.jpg"%index[0])
    if(index==0):continue
    save_image(img,"/kaggle/working/facadescd/test/%d.jpg"%(index[0]-1))

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
%cd pytorch-CycleGAN-and-pix2pix/
!pip install -r requirements.txt
!bash ./datasets/download_pix2pix_dataset.sh facades

!python train.py --dataroot /kaggle/input/facadescd/facadescd --name facades_pix2pix --model pix2pix --direction AtoB  --lr 0.001 --continue_train 

for i in range(4):
    !python test.py --dataroot /kaggle/input/facadescd/facadescd --direction AtoB --model pix2pix --name facades_pix2pix --results_dir /kaggle/working/test 

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("/kaggle/working/test/facades_pix2pix/test_latest/images/0_fake_B.png").convert('RGB')
plt.figure(figsize=(10,6),dpi=100)
plt.imshow(img)
plt.title("the image of prediction that 111s ")
plt.savefig("/kaggle/working/img/0_predit.jpg")
plt.show()

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("/kaggle/working/test/facades_pix2pix/test_latest/images/0_real_B.png").convert('RGB')
plt.figure(figsize=(10,6),dpi=100)
plt.imshow(img)
plt.title("the real image of 111s ")
plt.savefig("/kaggle/working/img/0_real.jpg")
plt.show()

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("/kaggle/working/test/facades_pix2pix/test_latest/images/125_fake_B.png").convert('RGB')
plt.figure(figsize=(10,6),dpi=100)
plt.imshow(img)
plt.title("the image of prediction that 126s ")
plt.savefig("/kaggle/working/img/0_predit.jpg")
plt.show()

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("/kaggle/working/test/facades_pix2pix/test_latest/images/125_real_B.png").convert('RGB')
plt.figure(figsize=(10,6),dpi=100)
plt.imshow(img)
plt.title("the real image of 126s ")
plt.savefig("/kaggle/working/img/0_real.jpg")
plt.show()