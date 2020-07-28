# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:05:45 2020

@author: Sophie
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator 


def seg_gen_train(img_path, msk_path, img_size, batch_size):
    datagenerator = ImageDataGenerator(rescale=1./255)
    gen_params = dict(target_size=img_size, class_mode=None, color_mode='grayscale', batch_size=batch_size)
    img_generator = datagenerator.flow_from_directory(img_path, **gen_params)
    msk_generator = datagenerator.flow_from_directory(msk_path, **gen_params)
    return zip(img_generator, msk_generator)

def seg_gen_test(img_path, msk_path, img_size, batch_size):
    datagenerator = ImageDataGenerator(rescale=1./255)
    gen_params = dict(target_size=img_size, class_mode=None, color_mode='grayscale', batch_size=batch_size)
    img_generator = datagenerator.flow_from_directory(img_path, **gen_params)
    msk_generator = datagenerator.flow_from_directory(msk_path, **gen_params)
    return zip(img_generator, msk_generator)
