# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:06:56 2020

@author: Sophie
"""

import os
import Unet
import datagenerator


data_dir = '/Users/Sophie/Documents/data/sub-all/slices'
data_dir_train = os.path.join(data_dir, 'training')
data_dir_train_image = os.path.join(data_dir_train, 'img')
data_dir_train_mask = os.path.join(data_dir_train, 'mask')
data_dir_test = os.path.join(data_dir, 'test')
data_dir_test_image = os.path.join(data_dir_test, 'img')
data_dir_test_mask = os.path.join(data_dir_test, 'mask')

BS_train = 32
BS_test = 32

img_h = 176
img_w = 256
img_size = (img_h, img_w)

num_train = 9504
num_test = 1728

epochs = 400
epoch_step_train = num_train // BS_train
epoch_step_test = num_test // BS_test

train_generator = datagenerator.seg_gen_train(data_dir_train_image, data_dir_train_mask, img_size, BS_train)
test_generator = datagenerator.seg_gen_test(data_dir_test_image, data_dir_test_mask, img_size, BS_test)


model = Unet.UNet(in_channels = 1 , out_channels = 1, n_levels = 4, initial_features = 32, n_blocks = 2, IMAGE_HEIGHT = img_h, IMAGE_WIDTH = img_w)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit_generator(generator=train_generator, 
                    steps_per_epoch=epoch_step_train, 
                    validation_data=test_generator, 
                    validation_steps=epoch_step_test,
                   epochs=epochs)

model.save(f'UNET-TissueSegmentation_{img_h}_{img_w}_{epochs}.h5')
