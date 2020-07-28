# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:15:51 2020

@author: Sophie
"""

from tensorflow.keras.models import load_model
import nibabel as nib
import os
import testfunctions


img_height = 176
img_width = 256
IMG_SIZE = (img_height, img_width)


model = load_model('/Users/Sophie/Downloads/UNET-GMSegmentation140_176_256.h5')

ImagePath = '/Users/Sophie/Documents/data/sub-all/anat/img/'
targetImagePath = os.path.join(ImagePath,'sub-22_T1w_brain.nii' )
MaskPath = '/Users/Sophie/Documents/data/sub-all/anat/mask2'
targetMaskPath = os.path.join(MaskPath, 'sub-22_T1w_brain_pve_1.nii.gz')

testfunctions.showprediction(model,targetImagePath, targetMaskPath, 80 )


dice_tab = []
iou_tab = []

dir_file_test = os.listdir(ImagePath)[21:]
mask_list = os.listdir(MaskPath)[21:]
    
for i in range (0, len(dir_file_test)):
    print(f'Starting the prediction of {dir_file_test[i]}: ')
    filename = f'/Users/Sophie/Documents/data/sub-all/anat/img/{dir_file_test[i]}'
    imgTarget = nib.load(filename).get_fdata()
    predImg = testfunctions.predictVolume(imgTarget, img_height, img_width, model)
    testfunctions.saveImg(predImg, f'prediction-{dir_file_test[i]}')
    

    targetMaskPath  = f'/Users/Sophie/Documents/data/sub-all/anat/mask2/{mask_list[i]}'
    imgMask = nib.load(targetMaskPath).get_fdata()
    iou_tab.append(testfunctions.iou(imgMask, predImg))
    dice_tab.append(testfunctions.dice(predImg, imgMask))
    
    print(f'Prediction of {dir_file_test[i]} DONE !')
    print(f'IoU : {iou_tab[i]}')
    print(f'Dice : {dice_tab[i]}')
        
