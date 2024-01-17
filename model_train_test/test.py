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
import seaborn as sns

from torchsummary import summary
import segmentation_models_pytorch as smp

from PIL import Image
import scipy.ndimage as ndi

from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = '../dataset_reduced_filled/test/'
TEST_MASK_PATH = '../dataset_reduced_filled/test_annot/'
TEST_OUTPUTS_PATH = '../dataset_reduced_filled/test_outputs/'

n_classes = 2
img_size = 256

def render_predictions(img, mask, shp=None):
    palette = sns.color_palette("bright", n_classes)
    cmap = sns.color_palette(palette, as_cmap=True)
    if shp != None:
        mask_cmap = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        mask_cmap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cmap_np = np.array(cmap)
    cmap_np = (cmap_np * 255).astype(np.uint8)
    for ix in range(mask.shape[0]):
        for iy in range(mask.shape[1]):
            if mask[ix, iy] != 0:
                mask_cmap[ix, iy, 0] = cmap_np[mask[ix, iy], 0]
                mask_cmap[ix, iy, 1] = cmap_np[mask[ix, iy], 1]
                mask_cmap[ix, iy, 2] = cmap_np[mask[ix, iy], 2]
    if shp != None:
        mask_cmap = cv2.resize(mask_cmap, (shp[1], shp[0]), cv2.INTER_NEAREST)
    mask_pil = Image.fromarray(mask_cmap, 'RGB')
    img_pil = Image.fromarray(img, 'RGB')

    mask_pil = mask_pil.convert("RGBA")
    img_pil = img_pil.convert("RGBA")
    new_img = Image.blend(img_pil, mask_pil, alpha=0.5)
    new_img.save("temp.png", "PNG")
    img_overlay = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    return img_overlay

def clean_mask(mask):
    output = np.zeros_like(mask)
    for muscle in range(1, 10): #excluding background, EPF_R (10), EPF_L (11)
        muscle_mask = mask == muscle
        if np.any(muscle_mask):
            labels, num_labels = ndi.label(muscle_mask)
            largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
            output[labels == largest_label] = muscle

    output = np.where(mask == 10, 10, output)
    output = np.where(mask == 11, 11, output)
    return output

if __name__ == "__main__":
    df_test = create_df(TEST_IMAGE_PATH)
    print('Total Images (test): ', len(df_test))
    X_test = df_test['id'].to_numpy()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    model = torch.load('model.pt')
    model.eval()

    t_test = A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST)
    test_set = ImageTestDataset(TEST_IMAGE_PATH, TEST_MASK_PATH, X_test, transform=t_test)

    for n in range(len(test_set)):
        image, mask = test_set[n]
        pred_mask, score = predict_image_mask_miou(model, image, mask)
        image_name = test_set.get_image_name(n).split('.')[0]
        #pred_mask_clean = clean_mask(pred_mask.cpu().numpy())
        pred_mask_clean = pred_mask.cpu().numpy()

        print(image_name)
        print('score: {}'.format(score))
        
        image_original = cv2.imread(TEST_IMAGE_PATH + image_name + '.png', cv2.IMREAD_COLOR)
        shape_original = image_original.shape
        image_original = cv2.resize(image_original, (shape_original[1], shape_original[0]))
        mask_original = cv2.imread(TEST_MASK_PATH + image_name + '.png', cv2.IMREAD_GRAYSCALE)
        painted_original = render_predictions(image_original, mask_original)
        painted_image = render_predictions(image_original, pred_mask_clean, shape_original)
        cv2.imwrite('{}{}_pred.png'.format(TEST_OUTPUTS_PATH, image_name), painted_image)
        cv2.imwrite('{}{}_gt.png'.format(TEST_OUTPUTS_PATH, image_name), painted_original)