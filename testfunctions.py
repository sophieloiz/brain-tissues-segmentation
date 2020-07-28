# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:15:51 2020

@author: Sophie
"""

import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt


def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

def predictVolume(inImg, height, width, model, toBin=True, SLICE_X = True, SLICE_Y = True, SLICE_Z = False):
    
    (x, y, z) = inImg.shape
    
    output_img_x = np.zeros((x, y, z))
    output_img_y = np.zeros((x, y, z))
    output_img_z = np.zeros((x, y, z))
    
    cpt = 0.0
    if SLICE_X:
        cpt += 1.0
        for i in range(x):
            img = scaleImg(inImg[i,:,:], height, width)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            output_img_x[i,:,:] = scaleImg(tmp, y, z)
    if SLICE_Y:
        cpt += 1.0
        for i in range(y):
            img = scaleImg(inImg[:,i,:],  height, width)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            output_img_y[:,i,:] = scaleImg(tmp, x, z)
    if SLICE_Z:
        cpt += 1.0
        for i in range(z):
            img = scaleImg(inImg[:,:,i],  height, width)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            output_img_z[:,:,i] = scaleImg(tmp, x, y)
    outImg = (output_img_x + output_img_y + output_img_z)/cpt
    
    if(toBin):
        outImg[outImg>0.5] = 1.0
        outImg[outImg<=0.5] = 0.0
    return outImg

def iou(mask, maskP):
    return np.sum(np.logical_and(mask, maskP))/np.sum(np.logical_or(mask, maskP))

def dice(predImg, imgMask):
    return np.sum(predImg[imgMask == 1])*2 / (np.sum(predImg) + np.sum(imgMask))

def saveImg(img, filename):
    return nib.Nifti1Image(img, np.eye(4)).to_filename(filename)

def image2array(img_path, mask_path):
    return nib.load(img_path).get_fdata(), nib.load(mask_path).get_fdata()

def prediction(model, img_path, mask_path, slice_index):
    imgTarget, imgMask = image2array(img_path, mask_path)  
    imgSlice = imgTarget[:,slice_index,:]
    imageInput = imgSlice[np.newaxis,:,:,np.newaxis]
    maskPredict = model.predict(imageInput)[0,:,:,0]
    maskSlice = imgMask[:,slice_index,:]
    return imgSlice, maskSlice, maskPredict

def showprediction(model, img_path, mask_path, slice_index):
    imgSlice, maskSlice, maskPredict = prediction(model, img_path, mask_path, slice_index)
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.title(title[0])
    plt.imshow(imgSlice, cmap='gray')
    plt.subplot(1,3,2)
    plt.title(title[1])
    plt.imshow(maskSlice, cmap='gray')
    plt.subplot(1,3,3)
    plt.title(title[2])
    plt.imshow(maskPredict, cmap = 'gray')
    plt.show()        
