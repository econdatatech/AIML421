#install what we need
import subprocess
import sys
import os
from subprocess import STDOUT, check_call

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install('torchsummary')
install('poutyne')
install('livelossplot') 
install('torchvision')
install('efficientnet_pytorch')



#housekeeping = import what we need
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchsummary import summary 
import poutyne
from poutyne.framework import Model,BestModelRestore,ModelCheckpoint,EarlyStopping  
from livelossplot import PlotLossesPoutyne # This module talks with Poutyne
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import copy
from imutils import paths
import cv2
from PIL import Image
import collections
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
import torch.utils.benchmark as benchmark

#set random seeds
np.random.seed(2342)
torch.manual_seed(2342)
poutyne.set_seeds(2342)

#check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print('The model will run on', device)

#remove images with wrong dimension
mydataset = ImageFolder("traindata/")
goodidx = [i for i in range(len(mydataset)) if Image.open(mydataset.imgs[i][0]).size == (300, 300) ]
mydataset = ImageFolder("traindata/",transform=transforms.ToTensor())
goodsubset = Subset(mydataset, goodidx)

#check class distribution
train_classes = [label for _, label in goodsubset]
print(collections.Counter(train_classes))

#define image normalisation values
mymean=[0.485, 0.456, 0.406]
mystd=[0.229, 0.224, 0.225]
    
#define various image transformations incl. augmentation
traintransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mymean,std=mystd)])
rtraintransform = transforms.Compose([transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)), transforms.RandomRotation(degrees=15),
                                      transforms.CenterCrop(size=300), transforms.ToTensor(), 
                                      transforms.Normalize(mean=mymean,std=mystd)])

vftraintransform = copy.deepcopy(traintransform)
vftraintransform.transforms.insert(0,transforms.RandomVerticalFlip(p=1))
hftraintransform = copy.deepcopy(traintransform)
hftraintransform.transforms.insert(0,transforms.RandomHorizontalFlip(p=1))
vhftraintransform = copy.deepcopy(traintransform)
vhftraintransform.transforms.insert(0,transforms.RandomHorizontalFlip(p=1))
vhftraintransform.transforms.insert(0,transforms.RandomVerticalFlip(p=1))
rvftraintransform = copy.deepcopy(rtraintransform)
rvftraintransform.transforms.insert(0,transforms.RandomVerticalFlip(p=1))
rhftraintransform = copy.deepcopy(rtraintransform)
rhftraintransform.transforms.insert(0,transforms.RandomHorizontalFlip(p=1))
rvhftraintransform = copy.deepcopy(rtraintransform)
rvhftraintransform.transforms.insert(0,transforms.RandomHorizontalFlip(p=1))
rvhftraintransform.transforms.insert(0,transforms.RandomVerticalFlip(p=1))

#define batch size
batch_size = 8

#define various image datasets based on the transformations and merge them
trainset = Subset(ImageFolder('traindata/',transform=traintransform), goodidx)
vftrainset = Subset(ImageFolder('traindata/', transform=vftraintransform), goodidx)
hftrainset = Subset(ImageFolder('traindata/', transform=hftraintransform), goodidx)
vfhftrainset = Subset(ImageFolder('traindata/', transform=vhftraintransform), goodidx)
rtrainset = Subset(ImageFolder('traindata/',transform=traintransform), goodidx)
rvftrainset = Subset(ImageFolder('traindata/', transform=rvftraintransform), goodidx)
rhftrainset = Subset(ImageFolder('traindata/', transform=rhftraintransform), goodidx)
rvfhftrainset = Subset(ImageFolder('traindata/', transform=rvhftraintransform), goodidx)
increased_dataset = torch.utils.data.ConcatDataset([trainset,vftrainset,hftrainset,vfhftrainset,rtrainset,rvftrainset,rhftrainset,rvfhftrainset])

#increased_dataset = torch.utils.data.ConcatDataset([trainset])

#make sure sure the training data set contains 80% original images + their augementations
#don't want to contaminate the validation data set with augmented versions of the training data set
num_train = len(trainset)
y = np.array([label for _, label in goodsubset])
X = list(range(num_train))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, valid_idx = next(sss.split(X, y))
new_list1 = train_idx+num_train
new_list2 = train_idx+2*num_train
new_list3 = train_idx+3*num_train
new_list4 = train_idx+4*num_train
new_list5 = train_idx+5*num_train
new_list6 = train_idx+6*num_train
new_list7 = train_idx+7*num_train
train_idx= np.concatenate((train_idx,new_list1,new_list2,new_list3,new_list4,new_list5,new_list6,new_list7))

#check how many images in the end
print('len(train_idx) ==> ', len(train_idx))
print('len(valid_idx) ==> ', len(valid_idx))

#define samples and loader
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
classes = ('cherry', 'strawberry', 'tomato')
trainloader = DataLoader(increased_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False,pin_memory=True)
valloader = DataLoader(increased_dataset, sampler=valid_sampler, batch_size=batch_size, shuffle=False,pin_memory=True)

#check class balances again
train_classes = [label for _, label in Subset(goodsubset,train_idx)]
print(collections.Counter(train_classes))
train_classes = [label for _, label in Subset(goodsubset,valid_idx)]
print(collections.Counter(train_classes))

#define helper function to make use of good stuff from poutyne
def better_poutyne_train(model_name, pytorch_model,plr=0.0001):
    callbacks = [
        # Save the latest weights 
        ModelCheckpoint(model_name + '_last_epoch.ckpt',temporary_filename='last_epoch.ckpt.tmp'),
        # EarlyStopping
        EarlyStopping(monitor='val_acc', patience=3, verbose=True, mode='max'),
        BestModelRestore(monitor='val_acc', mode='max')
    ]
    
    # Select the optimizer and the loss function 
    optimizer = optim.Adam(pytorch_model.parameters(), lr=plr)
    #optimizer = optim.SGD(pytorch_model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    # Poutyne Model
    model = Model(pytorch_model, optimizer, loss_function, batch_metrics=['accuracy'])
    # Send the 'Poutyne model' on GPU/CPU whichever is available 
    model.to(device)
    # Train

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
  
    model.fit_generator(trainloader, valloader, epochs=epochs, callbacks=callbacks)
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    
    print("training time in minutes: ")
    print(start.elapsed_time(end)/60000)

    # Test
    val_loss, val_acc, pred_y, true_y = model.evaluate_generator(valloader,return_pred=True,return_ground_truth=True)
    #print(f'Test:\n\tLoss: {val_loss: .3f}\n\tAccuracy: {val_acc: .3f}')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    return None
#define number of epochs
epochs=10
#get pretrained EfficientNet-B2
efficientnet = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3)
#finetune on my training data set
better_poutyne_train("trialXYZ",efficientnet)
#save the result
PATH = './verynewtry.pth'
torch.save(efficientnet.state_dict(), PATH)