import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../model_train_test/')
from dataset import *

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

import shutil
import nrrd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREDICT_IMAGE_PATH = './inputs/'
PREDICT_OUTPUTS_PATH = './outputs/'

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

if __name__ == "__main__":
    for root, dirs, files in os.walk(PREDICT_IMAGE_PATH):
        for f in files:
            if '.nrrd' in f:
                try:
                    shutil.rmtree('./temp_slices/')
                except:
                    print('Warning: temp directory not existing')
                os.mkdir('./temp_slices/')

                p_id = f.split('.')[0]

                data, header = nrrd.read(root + '/' + f)
                for k in range(data.shape[2]):
                    img = (data[:,:,k] - data[:,:,k].min()) / (data[:,:,k].max() - data[:,:,k].min()) * 255
                    img = np.round(img).astype(np.uint8)
                    cv2.imwrite('./temp_slices/{}_{}.png'.format(p_id, k), img)

                df_predict = create_df('./temp_slices/')
                print('Total Images ({}): {}'.format(p_id, len(df_predict)))
                X_predict = df_predict['id'].to_numpy()

                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]

                model = torch.load('../model_train_test/model.pt')
                model.eval()

                t_predict = A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST)
                predict_set = ImagePredictDataset('./temp_slices/', X_predict, transform=t_predict)

                try:
                    shutil.rmtree('{}{}'.format(PREDICT_OUTPUTS_PATH, p_id))
                except:
                    pass
                os.mkdir('{}{}'.format(PREDICT_OUTPUTS_PATH, p_id))

                for n in range(len(predict_set)):
                    image = predict_set[n]
                    pred_mask = predict_image(model, image)
                    image_name = predict_set.get_image_name(n).split('.')[0]
                    pred_mask_clean = pred_mask.cpu().numpy().astype(np.uint8)
                    
                    image_original = cv2.imread('./temp_slices/' + image_name + '.png', cv2.IMREAD_COLOR)
                    shape_original = image_original.shape
                    image_original = cv2.resize(image_original, (shape_original[1], shape_original[0]))
                    
                    painted_image = render_predictions(image_original, pred_mask_clean, shape_original)
                    cv2.imwrite('{}{}/{}_pred.png'.format(PREDICT_OUTPUTS_PATH, p_id, image_name), painted_image)

                    pred_mask_clean_resized = cv2.resize(pred_mask_clean, (shape_original[1], shape_original[0]), cv2.INTER_NEAREST)
                    pred_mask_clean_resized[pred_mask_clean_resized == 1] = 255
                    cv2.imwrite('{}{}/{}_mask.png'.format(PREDICT_OUTPUTS_PATH, p_id, image_name), pred_mask_clean_resized)
                    