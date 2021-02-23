import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from efficientnet_pytorch import EfficientNet
import cv2
import os
import random
import re
from fix_file import fixdir
fixdir()

def get_model(NUM_CLASSES):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    del model._fc
    for param in model.parameters():
        param.requires_grad = False
    # # # use the same head as the baseline notebook.
    model._fc = nn.Linear(1280, NUM_CLASSES)
    return model

num_class= 2
BATCH_SIZE =64
model = get_model(num_class)
device = torch.device("cuda" if torch.cuda.is_available() else cpu())
model = model.to(device)
model.load_state_dict(torch.load('model_efnetb0.pt'))


def get_mean_std(folder):
 
    means = torch.zeros(3)
    stds = torch.zeros(3)

    _data = datasets.ImageFolder(root = folder,transform = transforms.ToTensor())
    data_len = len(_data)

    for img, label in _data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= data_len
    stds /= data_len
    
    return means, stds

#test_means, test_stds = get_mean_std('data/test')
test_means, test_stds= (0.3604, 0.3993, 0.5377), (0.1755, 0.1858, 0.2364)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        
        return tuple_with_path

test_transforms= transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

root_path='data/test'
_data = ImageFolderWithPaths(root = root_path, transform = test_transforms)

test_loader = torch.utils.data.DataLoader(_data, 
                                batch_size = BATCH_SIZE)

batch_len = BATCH_SIZE
IMAGE =[]
LABEL = []
model.eval()
for data ,target,fname in test_loader:
    #fname =fname[0]
    
    data,target= data.to(device), target.to(device)
    output= model(data).cpu()
    pred = output.argmax(dim=1, keepdim=True)
    pred = torch.flatten(pred).numpy().tolist() 
    
    file_sz= len(fname)
    
    for j in range(file_sz):
        fn=int(re.search(r'\d+', fname[j]).group())
        IMAGE.append(fn)

    LABEL.extend(pred)
    
a=LABEL
a= [ "Fake" if item==0 else item for item in a]
a = ["Real" if item==1 else item for item in a]

sub= pd.DataFrame({'id': IMAGE, 'category': a})
df=sub.sort_values(by=['id']) 
file ='submission_efNet.csv'
df.to_csv(file, index=False)

print(file , "  saved...")