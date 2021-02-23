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
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        
        x = self.head0(x)
        self.y=x
        h = self.y.register_hook(self.activations_hook)  
        x= self.tail(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

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
    save_path ="Gradcam_Image/PyramidNemt/train"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)
    
if "test" in root_path: 
    save_path ="Gradcam_Image/PyramidNemt/test"
    _data = ImageFolderWithPaths(root = root_path, transform = train_transforms)

if "validation" in root_path: 
    save_path ="Gradcam_Image/PyramidNemt/val"
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
    net= pyrNet()
    net.eval()
    pred= net(img)
    idx= int(pred.argmax(dim=1)[0].cpu())
    
    if idx == 0 : 
        label= "fake"
    else:
        label= "real"
#    print(idx)
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

