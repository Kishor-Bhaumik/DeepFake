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
from PIL import Image, ImageOps, ImageEnhance
from efficientnet_pytorch import EfficientNet
from mpl_toolkits.axes_grid1 import ImageGrid
import pretrainedmodels
import cv2
import os
import random
from tqdm import tqdm
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
BATCH_SIZE = 32
num_class=2
pic_size =224

def get_model(NUM_CLASSES):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    del model._fc
    model._fc = nn.Linear(1280, NUM_CLASSES)
    return model

    
model = get_model(num_class)
model = model.to(device)
model.load_state_dict(torch.load('model_efnetb0.pt'))
    

class ENET(nn.Module):
    def __init__(self):
        
        super(ENET, self).__init__()

        self.ENet = model
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout=nn.Dropout(p=0.2, inplace=False)
        
        self.head0 = self.ENet.extract_features
        self._fc = self.ENet._fc

    def forward(self, x):
        
        x = self.head0(x)
        y=torch.flatten(x,1)
        x=self._avg_pooling(x)
        #register the hook
        x =x.flatten(start_dim=1)   
        x = self._dropout(x)
        x = self._fc(x)
        
        return x, y

    
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        
        return tuple_with_path





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

if "test" in FILE_INPUT:
    root_path ='data/test'
    
if "train" in FILE_INPUT:
    root_path ='data/train'
    
if "validation" in FILE_INPUT:
    root_path ='data/validation'
    

if "train" in root_path:
    save_path ="Tsne_Image/efficientNemt/train"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    
if "test" in root_path: 
    save_path ="Tsne_Image/efficientNemt/test"
    _data = ImageFolderWithPaths(root = root_path, transform = test_transforms)

if "validation" in root_path: 
    save_path ="Tsne_Image/efficientNemt/val"
    _data = ImageFolderWithPaths(root = root_path, transform = val_transforms)
    


data_loader = torch.utils.data.DataLoader(_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

efnet= ENET()
efnet.eval()



#test_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32)
test_predictions = []
#test_targets = []
test_embeddings = torch.zeros((0,  62720 ), dtype=torch.float32)

c= 0
for x, y, _ in data_loader:
    # _data.class_to_idx -> {'fake': 0, 'real': 1}
    x= x.to(device)
 
    logits, embeddings = efnet(x)
    preds = torch.argmax(logits, dim=1)
    test_predictions.extend(preds.detach().cpu().tolist())
#    test_targets.extend(y.detach().cpu().tolist())
    test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
#    test_imgs = torch.cat((test_imgs, x.detach().cpu()), 0)
    
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

plt.savefig(save_path+'/TsneEfnet.png')
plt.show()

D= time.time() - start_time
tt=str(datetime.timedelta(seconds = D)) 
print(" Time taken :" , tt)
