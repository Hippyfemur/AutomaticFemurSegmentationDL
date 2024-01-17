import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm
import ssl

from torchsummary import summary
import segmentation_models_pytorch as smp

from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMAGE_PATH = '../dataset_reduced_filled/train/'
TRAIN_MASK_PATH = '../dataset_reduced_filled/train_annot/'

n_classes = 2
img_size = 256

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context

    df_train = create_df(TRAIN_IMAGE_PATH)
    print('Total Images (train/val): ', len(df_train))

    X_train, X_val = train_test_split(df_train['id'].values, test_size=0.15, random_state=19)

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    t_train = A.Compose([A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                         A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                         A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                         A.GaussNoise()])

    t_val = A.Compose([A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                       A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), 
                       A.GridDistortion(p=0.2)])

    train_set = ImageDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std, t_train)
    val_set = ImageDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_val, mean, std, t_val)

    batch_size = 6

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)    

    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    #model = smp.PSPNet('resnet34', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.PAN('resnet34', encoder_weights='imagenet', classes=n_classes, activation=None)
    #model = smp.Unet('resnet101', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    
    max_lr = 1e-3
    epochs = 20
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    history = fit(epochs, model, train_loader, val_loader, criterion, optimizer, sched)
    torch.save(model, 'model.pt')