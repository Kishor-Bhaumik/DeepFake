import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
from efficientnet_pytorch import EfficientNet
from mpl_toolkits.axes_grid1 import ImageGrid
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable
import cv2
import os
import random
from tqdm import tqdm
from sklearn.utils import shuffle
import re
from fix_file import fixdir
fixdir()


BATCH_SIZE = 64
SEED=123
EPOCH = 50
pic_size= 224
device = torch.device("cuda" if torch.cuda.is_available() else cpu())
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
    
# train_means , train_stds = get_mean_std('data/train')
# val_means , val_stds = get_mean_std('data/validation')
# test_means, test_stds = get_mean_std('data/test')
train_means , train_stds = (0.3673, 0.4058, 0.5365),(0.1741, 0.1814, 0.2284)
val_means , val_stds     = (0.3440, 0.3894, 0.519),(0.1676, 0.1832, 0.2340)
test_means, test_stds= (0.3604, 0.3993, 0.5377), (0.1755, 0.1858, 0.2364)


train_transforms = transforms.Compose([
                           transforms.Scale(pic_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = train_means, 
                                                std = train_stds) ])

val_transforms = transforms.Compose([
                           transforms.Scale(pic_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = val_means, 
                                                std = val_stds) ])    

test_transforms = transforms.Compose([
                           transforms.Scale(pic_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = test_means, 
                                                std = test_means) ])


train_data = datasets.ImageFolder(root = 'data/train', 
                                  transform = train_transforms)
val_data = datasets.ImageFolder(root = 'data/validation', 
                                 transform = val_transforms)
test_data = datasets.ImageFolder(root = 'data/test', 
                                 transform = test_transforms)


train_loader = torch.utils.data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

valid_loader = torch.utils.data.DataLoader(val_data, shuffle= True,
                                 batch_size = BATCH_SIZE)

test_loader = torch.utils.data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)


model = ptcv_get_model("pyramidnet101_a360", pretrained=True)
num_ftrs = model.output.in_features
for param in model.parameters():
    param.requires_grad = False
    
model.output = nn.Linear(num_ftrs, 2)


model= model.to(device)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

set_seed(SEED)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

valid_loss_min = np.Inf

for epoch in range(EPOCH):
    train_loss = 0.0
    valid_loss = 0.0
    tcorrect = 0
    vcorrect = 0
    
    model.train()
    for data,target in train_loader:
        data,target= data.to(device), target.to(device)
        optimizer.zero_grad()
        output= model(data)
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        tcorrect += pred.eq(target.view_as(pred)).sum().item()
        
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*data.size(0)
        
        
    model.eval()
    for data,target in valid_loader:
        data,target= data.to(device), target.to(device)
        output= model(data)
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        vcorrect += pred.eq(target.view_as(pred)).sum().item()  

        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)

        
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    train_corrects= 100. * tcorrect / len(train_loader.sampler)
    val_corrects = 100. * vcorrect / len(valid_loader.sampler)
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(
        epoch, train_loss, valid_loss))
    print('Epoch: {} \tTraining acc: {:.2f} \tValidation acc: {:.2f} '.format(
        epoch, train_corrects, val_corrects))  
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_pyramid.pt')
        valid_loss_min = valid_loss
        
