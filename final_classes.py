# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:28:39 2020

@author: Sophie
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:29:27 2020

@author: Sophie
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:03:11 2020

@author: Sophie
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:04:26 2020

@author: Sophie
"""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
import os, glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras  
from tensorflow.keras.models import load_model
import tensorflow as tf



class SlicesConfig(object):

###########             SLICING           #########################
    
    Slice_X = True
    Slice_Y = False
    Slice_Z = False    
    dataInputPath ='/master/home/rto/abide4unet/'
    #dataInputPath = '/master/home/rto/oasis2/fsl-6.0.1/'
    dataOutputPath ='/master/home/rto/slices/train/'    
    img_name = '001_brain.nii.gz'
    mask_name = '001_brain_pve_0.nii.gz'
    Patches = True
    patch_size = 64
    batch_size = 64
  
############           TRAINING          #################################

    kernel_size=3
    n_levels = 4
    n_blocks=2
    pooling_size=2
    out_channels=1
    initial_features=32
    epochs = 100
    
    
    def displayConfiguration(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


        
class MRISlices(object):
    
    
    def __init__(self, config=SlicesConfig()):
        self.config = config 

        self.cpt = 0
        if self.config.Patches:

            self.IMAGE_H = self.config.patch_size
            self.IMAGE_W = self.config.patch_size   
            self.IMAGE_C = 1
        else:

            self.IMAGE_H = 256
            self.IMAGE_W = 256  
            self.IMAGE_C = 1
        self.model = self.UNet()   

        
    def readImageVolume(self, imgPath, normalization = False):
        if normalization : 
          img = MRISlices.normalizeImageIntensityRange(self, nib.load(imgPath).get_fdata())
        else : 
          img = nib.load(imgPath).get_fdata()
        return img
    
    def normalizeImageIntensityRange(self, img):
        H_MAX = np.max(img)
        H_MIN = np.min(img) 
        H_RANGE = H_MAX - H_MIN
        img[img < H_MIN] = H_MIN
        img[img > H_MAX] = H_MAX
        return (img - H_MIN)/H_RANGE
    
    
    def saveSlice(self, img, path, filename):
        img = np.uint8(img * 255)
        fout = os.path.join(path, f'{filename}.png')
        cv2.imwrite(fout, img)
    
        
    def saveSliceMask(self, img, path, filename):
        img[img>0.5]=1
        img[img<0.5]=0
        img = np.uint8(img * 255)
        fout = os.path.join(path, f'{filename}.png')
        cv2.imwrite(fout, img)
        
    
    
    def sliceSaveVolumePatch(self, vol, path, filename, patch_size):
        (xdim, ydim, zdim) = vol.shape
        cpt = 0
        if self.config.Slice_X:
            cpt += xdim
            if xdim >129:
                for i in range(80,130):
                    for k in range(patch_size,ydim-patch_size,patch_size):
                        for j in range(patch_size, zdim-patch_size, patch_size):
                          MRISlices.saveSlice(self,vol[i,k:k+patch_size,j:j+patch_size], path, filename+f'-slice{str(i)}_{k}_{j}_x')
                    
        if self.config.Slice_Y:
            cpt += ydim
            if ydim >129:
                for i in range(80,130):
                    for k in range(patch_size,ydim-patch_size,patch_size):
                        for j in range(patch_size, zdim-patch_size, patch_size):
                            MRISlices.saveSlice(self, vol[k:k+patch_size,i,j:j+patch_size], path, filename+f'-slice{str(i)}_{k}_{j}_y')
                    
    
        if self.config.Slice_Z:
            cpt += zdim
            if zdim >129:
                for i in range(80,130):
                    for k in range(patch_size,ydim-patch_size,patch_size):
                        for j in range(patch_size, zdim-patch_size, patch_size):
                            MRISlices.saveSlice(self, vol[k:k+patch_size,j:j+patch_size,i], path, filename+f'-slice{str(i)}_{k}_{j}_z')
                    
        return cpt
    
    def sliceSaveVolume(self, vol, path, filename):
        
        (xdim, ydim, zdim) = vol.shape
        cpt = 0
        if self.config.Slice_X:
            cpt += xdim
            if xdim >129:
                for i in range(80,130):
                    MRISlices.saveSlice(self, vol[i,:,:], path, filename+f'-slice{str(i)}_x')
    
        if self.config.Slice_Y:
            cpt += ydim
            if ydim >129:
                for i in range(80,130):
                    MRISlices.saveSlice(self, vol[:,i,:], path, filename+f'-slice{str(i)}_y')
    
        if self.config.Slice_Z:
            cpt += zdim
            if zdim >129:
                for i in range(80,130):
                    MRISlices.saveSlice(self, vol[:,:,i], path, filename+f'-slice{str(i)}_z')
        return cpt
    
    
    def sliceSaveVolumeM(self, vol, path, filename):
        (xdim, ydim, zdim) = vol.shape
        cpt = 0
        if self.config.Slice_X:
            cpt += xdim
            if xdim >129:
                for i in range(80,130):
                    MRISlices.saveSliceMask(self, vol[i,:,:], path, filename+f'-slice{str(i)}_x')
        if self.config.Slice_Y:
            cpt += ydim
            if ydim >129:
                for i in range(80,130):
                    MRISlices.saveSlice(self, vol[:,i,:], path, filename+f'-slice{str(i)}_y')
    
        if self.config.Slice_Z:
            cpt += zdim
            if zdim >129:
                for i in range(80,130):
                    MRISlices.saveSlice(self, vol[:,:,i], path, filename+f'-slice{str(i)}_z')
        return cpt
    
    
    def sliceSaveVolumeMPatch(self, vol, path, filename, patch_size):
        (xdim, ydim, zdim) = vol.shape
        cpt = 0
        if self.config.Slice_X:
            cpt += xdim
            if xdim >129:
                for i in range(80,130):
                  for k in range(patch_size,ydim-patch_size,patch_size):
                    for j in range(patch_size, zdim-patch_size, patch_size):
                      MRISlices.saveSliceMask(self, vol[i,k:k+patch_size,j:j+patch_size], path, filename+f'-slice{str(i)}_{k}_{j}_x')
    
        if self.config.Slice_Y:
            cpt += ydim
            for i in range(80,130):
                for k in range(patch_size,ydim-patch_size,patch_size):
                    for j in range(patch_size, zdim-patch_size, patch_size):
                      MRISlices.saveSliceMask(self, vol[k:k+patch_size,i,j:j+patch_size], path, filename+f'-slice{str(i)}_{k}_{j}_y')
    
    
        if self.config.Slice_Z:
            cpt += zdim
            for i in range(80,130):
                for k in range(patch_size,ydim-patch_size,patch_size):
                    for j in range(64, zdim-64, patch_size):
                      MRISlices.saveSliceMask(self, vol[k:k+patch_size,j:j+patch_size,i], path, filename+f'-slice{str(i)}_{k}_{j}_z')
    
        return cpt
    
    
    
    
    ##############################################################################
    ########SLICING FULL SLICES###################################################
    ##############################################################################
    
    def slicing(self):
        
        imageSliceOutput = os.path.join(self.config.dataOutputPath, 'img/img')
        maskSliceOutput = os.path.join(self.config.dataOutputPath, 'mask/img')
    
        for index, filename in enumerate(sorted(glob.iglob(self.config.dataInputPath+'*/' + self.config.img_name))):
            img = MRISlices.readImageVolume(self, filename, True)
        
            nSlices = MRISlices.sliceSaveVolume(self, img, imageSliceOutput, 'brain'+str(index))
            print(f'\n{filename}, {nSlices} slices created \n')
    
    
        for index, filename in enumerate(sorted(glob.iglob(self.config.dataInputPath +'*/' + self.config.mask_name))):
            img = MRISlices.readImageVolume(self, filename)
        
            nSlices = MRISlices.sliceSaveVolumeM(self, img, maskSliceOutput, 'brain'+str(index))
            print(f'\n{filename}, {nSlices} slices created \n')
    
    
    ##############################################################################
    ########SLICING PATCHES###################################################
    ##############################################################################
    
    def slicing_patches(self):
        
        imageSliceOutput = os.path.join(self.config.dataOutputPath, 'img/img')
        maskSliceOutput = os.path.join(self.config.dataOutputPath, 'mask/img')
        for index, filename in enumerate(sorted(glob.iglob(self.config.dataInputPath+'*/' +self.config.img_name))):
            img = MRISlices.readImageVolume(self, filename, True)
        
            nSlices = MRISlices.sliceSaveVolumePatch(self, img, imageSliceOutput, 'brain'+str(index),self.config.patch_size)
            print(f'\n{filename}, {nSlices} slices created \n')
        
        
        for index, filename in enumerate(sorted(glob.iglob(self.config.dataInputPath +'*/' +self.config.mask_name))):
            img = MRISlices.readImageVolume(self, filename)
        
            nSlices = MRISlices.sliceSaveVolumeMPatch(self, img, maskSliceOutput, 'brain'+str(index),self.config.patch_size)
            print(f'\n{filename}, {nSlices} slices created \n')


    ################################################################################
    ################################################################################
    ################################################################################

    def seg_gen_train(self):
        if self.config.Patches:
            img_size = (self.config.patch_size, self.config.patch_size)        
        else :
            img_size = (256, 256)
        img_path = os.path.join(self.config.dataOutputPath, 'img/')
        msk_path = os.path.join(self.config.dataOutputPath, 'mask/')
        self.cpt = len(os.listdir(img_path+'img/'))
        datagenerator = ImageDataGenerator(rescale=1./255)
        gen_params = dict(target_size=img_size, class_mode=None, color_mode='grayscale', batch_size=self.config.batch_size, seed = 909 )
        img_generator = datagenerator.flow_from_directory(img_path, **gen_params)
        msk_generator = datagenerator.flow_from_directory(msk_path, **gen_params)
        return zip(img_generator, msk_generator)

  
    def UNet(self):
        inputs = keras.layers.Input(shape=(self.IMAGE_H, self.IMAGE_W, self.IMAGE_C))    
        x = inputs
        convpars = dict(kernel_size=3, activation='relu', padding='same')
            #downsampling (context information)
        skips = {}
        for level in range(self.config.n_levels):
            for _ in range(self.config.n_blocks):
                x = keras.layers.Conv2D(self.config.initial_features * 2 ** level, **convpars)(x) #convblock
            if level < self.config.n_levels - 1:
                skips[level] = x #skipconnection
                x = keras.layers.MaxPool2D(self.config.pooling_size)(x) #Maxpooling
                
        # upsampling (localization information)
        for level in reversed(range(self.config.n_levels-1)): #reversed pour commencer par la fin
            x = keras.layers.Conv2DTranspose(self.config.initial_features * 2 ** level, strides=self.config.pooling_size, **convpars)(x)
            x = keras.layers.Concatenate()([x, skips[level]]) #Concaténation
            for _ in range(self.config.n_blocks):
                x = keras.layers.Conv2D(self.config.initial_features * 2 ** level, **convpars)(x)
                
        # output
        activation = 'sigmoid' if self.config.out_channels == 1 else 'softmax'
        x = keras.layers.Conv2D(self.config.out_channels, kernel_size=1, activation=activation, padding='same')(x)

        model = keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{self.config.n_levels}-F{self.config.initial_features}')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, train_gen,name):
        epoch_step_train = self.cpt //self.config.batch_size        
        with tf.device('/device:GPU:0'):
          self.model.fit(train_gen, steps_per_epoch=epoch_step_train,epochs=self.config.epochs)
        self.model.save(name)
 
class PredictConfig(object):

    SLICE_X = True
    SLICE_Y = False
    SLICE_Z = False    
    
    T1W = False
    targetImagePath = '/Users/Sophie/Downloads/orangutan_Orangoutan_80bb_v2_testEB.nii.gz' 
    targetOutput = '/Users/Sophie/Downloads/'
    
    def displayConfiguration(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



#configuration = SlicesConfig()
#configuration.displayConfiguration()
#mrislices = MRISlices()
#mrislices.slicing()
#datagen = mrislices.seg_gen_train()
#mrislices.train(datagen, 'test.h5')
        

class Predict(object):

   def __init__(self, config=PredictConfig()):
        self.config = config 

        self.cpt = 0
        if self.config.T1W:
                    
            self.model_pve0_patchs = load_model('/Users/Sophie/Desktop/First_pve0_15_T2_patches.h5')
            self.model_pve1_patchs = load_model('/Users/Sophie/Desktop/First_pve1_15_T2_patches.h5')
            self.model_pve2_patchs = load_model('/Users/Sophie/Desktop/First_pve2_15_T2_patches.h5')
            
        else:

            self.model_pve0_patchs = load_model('/Users/Sophie/Desktop/First_pve0_35_T1_patches.h5')
            self.model_pve1_patchs = load_model('/Users/Sophie/Desktop/First_pve1_35_T1_patches.h5')
            self.model_pve2_patchs = load_model('/Users/Sophie/Desktop/First_pve2_10_T1_patch.h5')
    
   def normalizeImageIntensityRange(self, img):
        H_MAX = np.max(img)
        H_MIN = np.min(img) 
        H_RANGE = H_MAX - H_MIN
        img[img < H_MIN] = H_MIN
        img[img > H_MAX] = H_MAX
        return (img - H_MIN)/H_RANGE

   def scaleImg(self, img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    
   def predict_slices(self, sliceIndex):
        
        imgTargetNii = nib.load(self.config.targetImagePath)    
        imgTarget = imgTargetNii.get_fdata()
        imgTarget = Predict.normalizeImageIntensityRange(self, imgTarget)
        imgSlice = imgTarget[sliceIndex,:,:]      
        imgSliceScaled = Predict.scaleImg(self, imgSlice,256,256)    
        imageInput = imgSliceScaled[np.newaxis,:,:,np.newaxis]
        
        maskPredict = self.model_pve2_patchs.predict(imageInput)
        maskPredict1 = self.model_pve1_patchs.predict(imageInput)
        maskPredict0 = self.model_pve0_patchs.predict(imageInput)  
        
        return imgSliceScaled, maskPredict0, maskPredict1, maskPredict
                        
    
   def show_prediction(self, sliceIndex):
        
        rgb = np.zeros((256,256,3))
        imgSliceScaled, maskPredict0, maskPredict1, maskPredict = Predict.predict_slices(self, sliceIndex)
        rgb[maskPredict0[0,:,:,0]>0.5]=(1,0,0)
        rgb[maskPredict1[0,:,:,0]>0.5]=(0,1,0)
        rgb[maskPredict[0,:,:,0]>0.5]=(0,0,1)
        
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(imgSliceScaled, cmap = 'gray')
        plt.subplot(1,2,2)
        plt.imshow(rgb)
        plt.show()
        
   def predict_full_volume(self):
        
        imgTargetNii = nib.load(self.config.targetImagePath)    
        imgTarget = imgTargetNii.get_fdata()
        inImg = Predict.normalizeImageIntensityRange(self, imgTarget)
        (xMax, yMax, zMax) = inImg.shape
        
        outImgX = np.zeros((xMax, yMax, zMax))
        outImgY = np.zeros((xMax, yMax, zMax))
        outImgZ = np.zeros((xMax, yMax, zMax))

        outImgX1 = np.zeros((xMax, yMax, zMax))
        outImgY1 = np.zeros((xMax, yMax, zMax))
        outImgZ1 = np.zeros((xMax, yMax, zMax))

        outImgX2 = np.zeros((xMax, yMax, zMax))
        outImgY2 = np.zeros((xMax, yMax, zMax))
        outImgZ2 = np.zeros((xMax, yMax, zMax))        
        cnt = 0.0
        if self.config.SLICE_X:
            cnt += 1.0
            for i in range(xMax):
                img = Predict.scaleImg(self, inImg[i,:,:], 256, 256)[np.newaxis,:,:,np.newaxis]
                tmp = self.model_pve0_patchs.predict(img)[0,:,:,0]
                outImgX[i,:,:] = Predict.scaleImg(self, tmp, yMax, zMax)
                tmp1 = self.model_pve1_patchs.predict(img)[0,:,:,0]
                outImgX1[i,:,:] = Predict.scaleImg(self,tmp1, yMax, zMax)
                tmp2 = self.model_pve2_patchs.predict(img)[0,:,:,0]
                outImgX2[i,:,:] = Predict.scaleImg(self,tmp2, yMax, zMax)
                
        if self.config.SLICE_Y:
            cnt += 1.0
            for i in range(yMax):
                img = Predict.scaleImg(self, inImg[:,i,:], 256, 256)[np.newaxis,:,:,np.newaxis]
                tmp = self.model_pve0_patchs.predict(img)[0,:,:,0]
                outImgY[:,i,:] = Predict.scaleImg(self,tmp, xMax, zMax)
                tmp1 = self.model_pve1_patchs.predict(img)[0,:,:,0]
                outImgY1[:,i,:] = Predict.scaleImg(self,tmp1, xMax, zMax)
                tmp2 = self.model_pve2_patchs.predict(img)[0,:,:,0]
                outImgY2[:,i,:] = Predict.scaleImg(self,tmp2, xMax, yMax)
        if self.config.SLICE_Z:
            cnt += 1.0
            for i in range(zMax):
                img = Predict.scaleImg(self, inImg[:,:,i], 256, 256)[np.newaxis,:,:,np.newaxis]
                tmp = self.model_pve0_patchs.predict(img)[0,:,:,0]
                outImgZ[:,:,i] = Predict.scaleImg(self,tmp, xMax, yMax)
                tmp1 = self.model_pve1_patchs.predict(img)[0,:,:,0]
                outImgZ1[:,:,i] = Predict.scaleImg(self,tmp1, xMax, yMax)
                tmp2 = self.model_pve2_patchs.predict(img)[0,:,:,0]
                outImgZ2[:,:,i] = Predict.scaleImg(self,tmp2, xMax, yMax) 
                
        outImg = (outImgX + outImgY + outImgZ)/cnt
        outImg1 = (outImgX1 + outImgY1 + outImgZ1)/cnt
        outImg2 = (outImgX2 + outImgY2 + outImgZ2)/cnt
        
        outImg[outImg>0.5] = 1.0
        outImg[outImg<=0.5] = 0.0
        
        outImg1[outImg1>0.5] = 2.0
        outImg1[outImg1<=0.5] = 0.0
        
        outImg2[outImg2>0.5] = 3.0
        outImg2[outImg2<=0.5] = 0.0
        

        nib.Nifti1Image(outImg, imgTargetNii.affine).to_filename(self.config.targetOutput + 'pred_001_brain_pve_0.nii')
           
        nib.Nifti1Image(outImg1, imgTargetNii.affine).to_filename(self.config.targetOutput + 'pred_001_brain_pve_1.nii')
        
        nib.Nifti1Image(outImg2, imgTargetNii.affine).to_filename(self.config.targetOutput + 'pred_001_brain_pve_2.nii')
        
        return outImg, outImg1, outImg2


configuration = PredictConfig() 
configuration.displayConfiguration()

prediction = Predict()
#prediction.show_prediction(100)  
prediction.predict_full_volume()   

  
