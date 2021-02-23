import time
start_time = time.time()
import argparse
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.nn as nn
import imutils
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorchcv.model_provider import get_model as ptcv_get_model
import cv2
import os
import random

from pathlib import Path
import glob
import datetime
from sklearn.manifold import TSNE
from matplotlib import cm
from fix_file import fixdir
fixdir()


parser = argparse.ArgumentParser(description='T-SNE Visualization')

parser.add_argument('--choose_file', type=str, default='test',
                        help=' choose image from train or test or validation ')    

args = parser.parse_args()

FILE_INPUT= args.choose_file
device = torch.device("cuda" if torch.cuda.is_available() else cpu())
BATCH_SIZE = 16
num_class=2

model = ptcv_get_model("pyramidnet101_a360", pretrained=True)
num_ftrs = model.output.in_features
model.output = nn.Linear(num_ftrs, num_class)
model.load_state_dict(torch.load('model_pyramid.pt'))
model= model.to(device)


class pyrNet(nn.Module):
    def __init__(self):
        
        super(pyrNet, self).__init__()

        self.pyrnet= model        
        self.head0 = self.pyrnet.features[:-1]
        self.tail = self.pyrnet.features[-1:]
        self.output = self.pyrnet.output

    def forward(self, x):
        
        x = self.head0(x)
        y= torch.flatten(x,1)
        x= self.tail(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x,y

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        
        return tuple_with_path

train_transforms= transforms.Compose([transforms.CenterCrop((224,224)), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if "test" in FILE_INPUT:
    root_path ='data/test'
    
if "train" in FILE_INPUT:
    root_path ='data/train'
    
if "validation" in FILE_INPUT:
    root_path ='data/validation'
    

if "train" in root_path:
    save_path ="Tsne_Image/PyramidNet/train"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    
if "test" in root_path: 
    save_path ="Tsne_Image/PyramidNet/test"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)

if "validation" in root_path: 
    save_path ="Tsne_Image/PyramidNet/val"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    


data_loader = torch.utils.data.DataLoader(_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

net= pyrNet()
net.eval()

#test_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32)
test_predictions = []
#test_targets = []
test_embeddings = torch.zeros((0,83104), dtype=torch.float32)

c= 0
for x, y, _ in data_loader:
    # _data.class_to_idx -> {'fake': 0, 'real': 1}
    x= x.to(device)
 
    logits, embeddings = net(x)
    preds = torch.argmax(logits, dim=1)
    test_predictions.extend(preds.detach().cpu().tolist())
    #test_targets.extend(y.detach().cpu().tolist())
    test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
    #test_imgs = torch.cat((test_imgs, x.detach().cpu()), 0)
    
#     c+=1
#     if c==10 : break
    
#test_imgs = np.array(test_imgs)
test_embeddings = np.array(test_embeddings)
#test_targets = np.array(test_targets)
test_predictions = np.array(test_predictions)


tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(16,8))
num_categories = 2
for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4),
               s=80,label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)

ax.set_xlabel('fake: 0 , real: 1')

Path(save_path).mkdir(parents=True, exist_ok=True)
files = glob.glob(save_path+"/*")
for f in files: os.remove(f)

plt.savefig(save_path+'/TsnePyramidNet.png')
plt.show()

D= time.time() - start_time
tt=str(datetime.timedelta(seconds = D)) 
print("--- %s Time taken ---" % (tt))
