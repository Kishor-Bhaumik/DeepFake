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
from fix_file import fixdir
fixdir()


parser = argparse.ArgumentParser(description='GradCam Visualization')

parser.add_argument('--choose_file', type=str, default='train',
                        help=' choose image from train or test ')    
parser.add_argument('--Num_img', type=int, default =5,
                          help='Number of images')   
args = parser.parse_args()

FILE_INPUT= args.choose_file
NUM_IMAGE= args.Num_img
device = torch.device("cuda" if torch.cuda.is_available() else cpu())

num_class=2

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

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        
        x = self.head0(x)
        self.y= x
        x=self._avg_pooling(x)
        #register the hook
        x =x.flatten(start_dim=1)
        h = self.y.register_hook(self.activations_hook)       
        x = self._dropout(x)
        x = self._fc(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.head0(x)
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
    save_path ="Gradcam_Image/efficientNemt/train"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    
if "test" in root_path: 
    save_path ="Gradcam_Image/efficientNemt/test"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)

if "validation" in root_path: 
    save_path ="Gradcam_Image/efficientNemt/val"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    

BATCH_SIZE = 1

data_loader = torch.utils.data.DataLoader(_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)


def overlay_heatmap(heatmp, image, alpha=0.5,colormap=cv2.COLORMAP_VIRIDIS):
    
    heatmp = np.uint8(255 * heatmp)
    heatmp = cv2.applyColorMap(heatmp, colormap)
    output = cv2.addWeighted(image, alpha, heatmp, 1 - alpha, 0)
    
    return (heatmp, output)

def save_image(path,save,heatmap,label):

    img = cv2.imread(path)
    Heatmap =heatmap.numpy()
    
    Heatmap = cv2.resize(src=Heatmap, dsize=(img.shape[1], img.shape[0]))
    (Heatmap, output) = overlay_heatmap(Heatmap, img, alpha=0.5)
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.vstack([img, Heatmap, output])
    output = imutils.resize(output, height=1000)
    fname = save+ "/" + path.rsplit('/', 1)[-1]
    cv2.imwrite(fname, output)
    
    
Path(save_path).mkdir(parents=True, exist_ok=True)

files = glob.glob(save_path+"/*")
for f in files: os.remove(f)

count=0
for img , target, fname in data_loader:
    
    
    # _data.class_to_idx -> {'fake': 0, 'real': 1}
    
    fname =fname[0]
    img= img.to(device)
    efnet= ENET()
    efnet.eval()
    pred= efnet(img)
    idx= int(pred.argmax(dim=1)[0].cpu())
    
    if idx == 0 : 
        label= "fake"
    else:
        label= "real"

        
    pred[:, idx].backward()

    gradients = efnet.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = efnet.get_activations(img).detach()
    R= activations.size(1)
    # weight the channels by corresponding gradients


    for i in range(R):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    #plt.matshow(heatmap.squeeze())
    
    save_image(fname, save_path,heatmap,label)
    
    count+=1
    if count == NUM_IMAGE : break

