import os
import numpy as np
from shutil import copyfile
import random
import cv2

pth = './images_filled/'
pth_out = './dataset_reduced_filled/'

prob_train = 0.8
prob_included = 0.05

patients_train_test = np.zeros(101, dtype=np.uint8)

if __name__ == "__main__":
    for root, dirs, files in os.walk(pth):
        for f in files:
            if '_s.png' in f:
                p_id = int(f.split('_')[1].split('Pat')[1])
                n_slice = int((f.split('_')[2]).split('.')[0])
                
                inclusion = random.uniform(0, 1)
                if inclusion < prob_included:
                    print(p_id)

                    if patients_train_test[p_id - 1] == 0: # not allocated
                        allocation = random.uniform(0, 1)
                        if allocation < prob_train:
                            patients_train_test[p_id - 1] = 1 # train
                        else:
                            patients_train_test[p_id - 1] = 2 # test

                    img_s = cv2.imread(root + '/' + f, cv2.IMREAD_GRAYSCALE)
                    img_s[img_s == 255] = 1

                    if patients_train_test[p_id - 1] == 1:
                        filename_s = pth_out + 'train_annot/{}_{}.png'.format(p_id, n_slice)
                    else:
                        filename_s = pth_out + 'test_annot/{}_{}.png'.format(p_id, n_slice)
                    cv2.imwrite(filename_s, img_s)

                    f_img = f.replace('_s.png', '.png')
                    if patients_train_test[p_id - 1] == 1:
                        filename_img = pth_out + 'train/{}_{}.png'.format(p_id, n_slice)
                    else:
                        filename_img = pth_out + 'test/{}_{}.png'.format(p_id, n_slice)

                    copyfile(root + '/' + f_img, filename_img)
                

                    