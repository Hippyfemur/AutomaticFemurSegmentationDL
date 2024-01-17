import os
import nrrd
import cv2
import numpy as np
import scipy
import scipy.ndimage
import stl_reader
import stltovoxel
from stl import mesh
import pyvista as pv
import SimpleITK as sitk
import trimesh
from copy import deepcopy

data_pth = './data_original/'
out_image_pth = './images_filled/'

def save_slices(data, p_id, max_slices = 0):
    if max_slices == 0:
        n_slices = data.shape[2]
    else:
        n_slices = max_slices
    for islice in range(n_slices):
        img = (data[:,:,islice] - data[:,:,islice].min()) / (data[:,:,islice].max() - data[:,:,islice].min()) * 255
        img = np.round(img).astype(np.uint8)
        cv2.imwrite(out_image_pth + p_id + '_s{}.png'.format(islice), img)
    return

def voxelize_stl(f_stl, f_nrrd, p_id, max_slices = 0):
    stl_mesh = trimesh.load(f_stl)
    stl_voxels = stl_mesh.voxelized(pitch=1.).fill(method="holes")
    print('watertight {}'.format(stl_mesh.is_watertight))

    seg_data = stl_voxels.matrix
    seg_size = stl_voxels.pitch
    seg_origin = stl_voxels.transform[:3, 3]
    print('seg_size: {}'.format(seg_size))
    print('seg_origin: {}'.format(seg_origin))

    data, header = nrrd.read(f_nrrd)

    voxels_remapped = np.zeros(data.shape, dtype=np.uint8)
    spacing = header['space directions']
    
    for i in range(data.shape[0]):
        print('{}/{}'.format(i, data.shape[0]))
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                origin = header['space origin']
                x = origin[0] + i * spacing[0][0]
                y = origin[1] + j * spacing[1][1]
                z = origin[2] + k * spacing[2][2]

                sx = int((x - seg_origin[0]) / seg_size[0])
                sy = int((y - seg_origin[1]) / seg_size[1]) 
                sz = int((z - seg_origin[2]) / seg_size[2])

                if (sx >= 0) and (sx < seg_data.shape[0] -1) and (sy >= 0) and (sy < seg_data.shape[1] -1) and (sz >= 0) and (sz < seg_data.shape[2] -1):
                    if seg_data[sx, sy, sz] != 0:
                        voxels_remapped[i, j, k] = 255

    for k in range(data.shape[2]):
        img = (data[:,:,k] - data[:,:,k].min()) / (data[:,:,k].max() - data[:,:,k].min()) * 255
        img = np.round(img).astype(np.uint8)
        cv2.imwrite(out_image_pth + 'ct_{}_{}.png'.format(p_id, k), img)

        #segmentation_filled = fill_segmentation(voxels_remapped[:,:,k])
        #cv2.imwrite(out_image_pth + 'ct_{}_{}_s.png'.format(p_id, k), segmentation_filled)

        cv2.imwrite(out_image_pth + 'ct_{}_{}_s.png'.format(p_id, k), voxels_remapped[:,:,k])
    return

if __name__ == "__main__":
    for root, dirs, files in os.walk(data_pth):
        for f in files:
            if '.nrrd' in f:
                p_id = root.split('/')[-1]
                print(p_id)

                data, header = nrrd.read(root + '/' + f)
                print(data.shape)
                print(header)
                #save_slices(data, p_id, max_slices = 2)

                voxelize_stl(root + '/' + f.replace('.nrrd', '.stl'), root + '/' + f, p_id, max_slices = 2)

