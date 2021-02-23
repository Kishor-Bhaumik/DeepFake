import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pretrainedmodels
import cv2
import os
from pathlib import Path
import random
import re
import glob
import argparse
import imutils
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

model_name = 'xception' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name]()
num_ftrs = model.last_linear.in_features
model.last_linear = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load('model_xception.pt'))


class xcep(nn.Module):
    def __init__(self):
        
        super(xcep, self).__init__()

        self.xcepnet= model        
        self.head0 = self.xcepnet.features
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = self.xcepnet.last_linear
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        
        x = self.head0(x)
        self.y=x
        h = self.y.register_hook(self.activations_hook)  
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

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
    save_path ="Gradcam_Image/XceptionNemt/train"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    
if "test" in root_path: 
    save_path ="Gradcam_Image/XcptionNemt/test"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)

if "validation" in root_path: 
    save_path ="Gradcam_Image/XceptionNemt/val"
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
    net= xcep()
    net.eval()
    pred= net(img)
    idx= int(pred.argmax(dim=1)[0].cpu())
    
    if idx == 0 : 
        label= "fake"
    else:
        label= "real"

    pred[:, idx].backward()

    gradients = net.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = net.get_activations(img).detach()
    R= activations.size(1)
    # weight the channels by corresponding gradients

    for i in range(R):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    #plt.matshow(heatmap.squeeze())
    save_image(fname, save_path,heatmap,label)
    
    count+=1
    if count == NUM_IMAGE : break

