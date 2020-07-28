# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os, glob
import nibabel as nib
import numpy as np
import cv2


dataInputPath ='/Users/Sophie/Documents/data/sub-all/anat/'
imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'mask2/')

dataOutputPath = '/Users/Sophie/Documents/data/sub-light/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/')
maskSliceOutput = os.path.join(dataOutputPath, 'mask/')

Slice_X = True
Slice_Y = True
Slice_Z = False


def readImageVolume(imgPath):
    return nib.load(imgPath).get_fdata()
    
        
def saveSlice(img, path, filename):
    img = np.uint8(img * 255)
    output_file = os.path.join(path, f'{filename}.png')
    cv2.imwrite(output_file, img)
    print(f'Slice saved: {output_file}')
    
  
def Volume2Slices(vol, path, filename):
    (xdim, ydim, zdim) = vol.shape
    cpt = 0
    if Slice_X:
        cpt += xdim
        for i in range(xdim):
            saveSlice(vol[i,:,:], path, filename+f'-slice{str(i)}_x')
            
    if Slice_Y:
        cpt += ydim
        for i in range(ydim):
            saveSlice(vol[:,i,:], path, filename+f'-slice{str(i)}_y')
            
    if Slice_Z:
        cpt += zdim
        for i in range(zdim):
            saveSlice(vol[:,:,i], path, filename+f'-slice{str(i)}_z')
    return cpt



for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii'))):
    img = readImageVolume(filename)
    nSlices = Volume2Slices(img, imageSliceOutput, 'brain'+str(index))
    print(f'\n{filename}, {nSlices} slices created \n')
    

for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii.gz'))):
    img = readImageVolume(filename)
    nSlices = Volume2Slices(img, maskSliceOutput, 'brain'+str(index))
    print(f'\n{filename}, {nSlices} slices created \n')
