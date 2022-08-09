#install what we need
import subprocess
import sys

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
from PIL import Image
import collections
import torch.utils.benchmark as benchmark
from torch.utils.data import Dataset
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print('The model will run on', device)

#define normalisation transform
mymean=[0.485, 0.456, 0.406]
mystd=[0.229, 0.224, 0.225]
testtransform = transforms.Compose([transforms.Resize(size=[300,300]),transforms.ToTensor(), transforms.Normalize(mean=mymean,std=mystd)])

#define testset and image/dataloader
testset =  ImageFolder('testdata/',transform=testtransform)
batch_size=8
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,pin_memory=True)

#get efficientnet
model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=3)

#load state
PATH = './model.pth'
model.load_state_dict(torch.load(PATH))

#define poutyne model
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()
pmodel = Model(model, optimizer, loss_function, batch_metrics=['accuracy'],device=device)

#get predictions
test_loss, test_acc, pred_y, true_y = pmodel.evaluate_generator(testloader,return_pred=True,return_ground_truth=True)

#could run this to get the class prediction 
#preds_classes = np.argmax(pred_y, axis=-1)
#preds_classes