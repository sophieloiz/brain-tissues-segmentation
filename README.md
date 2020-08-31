# Brain Tissues Segmentation

## Introduction 
The purpose of this project is to develop deep learning approaches for the segmentation of brain tissues. These segmentations are useful for measuring and visualizing anatomical structures, but also to analyze brain changes in case of diseases like Alzheimer. Today different automatic segmentations are available thanks to FAST (FSL), Freesurfer and ANTS. But these approaches are often inaccurate and require additional manual segmentations which are both time consuming and challenging. 

## Dataset

For this project, we used more than 1000 MRIs from a public dataset : the Autism Brain Imaging Data Exchange (ABIDE), which is composed of functional and structural brain imaging data collected from laboratories around the world. This dataset was created to accelerate the understanding of the neural bases of autism. It provides us for each subject : the original MRI, the brain extracted from the skull and the segmentation into 3 tissues (white matter, gray matter and core white matter). 

## First Approach : UNet Implementation

### Preprocessing

The first UNet implemented took 2D images as input. So, we needed to slice the 3D volume images of our dataset.  

This is an example of a single slice of the input image and the associated masks : 

![Image Mask](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/true_masks.png)


This function slices the whole volume images into different 2D slices :



```javascript
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
    
```

An other important step was to normalize each slice of the MRI in order to make the network works for every MRI. 

Here is an example of two histograms of a slice. On the left we can observe the histogram before normalization and on the right the result after normalization. 

![Image Mask](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/histo.png)

Thus, every pixel has a value between 0 and 1.
### Model 

The U-net is a convolutional network architecture used for fast and precise segmentation of images. This is a very popular architecture in biomedical images. 

The architecture contains two paths:
* a contraction path => encoder
* an expanding path => decoder
 
 The **encoder** is used to capture the context in the image, whereas the **decoder** will enable precise localization. 



![Image Mask](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/u-net-architecture.png)



This is the code relative to this architecture: 

```javascript
def UNet(in_channels, out_channels, n_levels, initial_features, n_blocks, IMAGE_HEIGHT, IMAGE_WIDTH):
   
    inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    
    x = inputs
    
    skips_connections = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = layers.Conv2D(initial_features * 2 ** level, kernel_size=3, activation='relu', padding='same')(x)
        if level < n_levels - 1:
            skips_connections[level] = x 
            x = layers.MaxPool2D(2)(x) 
            
    for level in reversed(range(n_levels-1)): 
        x = layers.Conv2DTranspose(initial_features * 2 ** level, strides=2, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.Concatenate()([x, skips_connections[level]]) 
        for _ in range(n_blocks):
            x = layers.Conv2D(initial_features * 2 ** level, kernel_size=3, activation='relu', padding='same')(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-Level{n_levels}-Features{initial_features}')
```

### Train 

To train the model, **Adam** was used as an optimizer and **binary crossentropy** as loss function.

```javascript

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_generator, 
                    steps_per_epoch=epoch_step_train, 
                    validation_data=test_generator, 
                    validation_steps=epoch_step_test,
                   epochs=epochs)
```
To evaluate the model we used two of the most common metrics for semantic segmentation : 

* The Intersection-Over-Union (IoU)
* The Dice Coefficient (F1 Score)

In order to evaluate the **F1 Score**, a function was created : 

```javascript
def get_f1(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
```

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results_loss_accuracy_pve1.png)

### Results

Some examples of the results of the network : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results.png)

Let's focus on one 2D-Slice : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results_unet%20-.png)


For this example the dice score and the IoU were calculated for each tissue : 

---| Dice Score| IoU 
--- | --- | ---
WM | 0.846060| 0.733194 
GM | 0.947726| 0.900646 
CSF | 0.84289| 0.728459 

## Second Approach : 2D Patches Segmentation with a UNet Architecture

Neuroscientists who want to work on non-human MRI can't use standard segmentation tools like **FAST FSL** to segmentate brain tissues like white and grey matter. In order to avoid manual Segmentation which is booth challenging and time consuming, we try to developp a deep learning approach by using small patches. The idea was instead of training the network on complete slices, to train the network on pieces of the image so that we can use it for non-human data.  

### Preprocessing 

The UNet implemented took 2D patches of size : 64x64 as input. Thus, we had to generate several patches per MRI slice. 

A first approach was to resize each slice to (256x256) and then to divide it into 16 patches.

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/Input-patch.png)

Only patches with information was used to train the model. So we removed all patches where the maximum value was 0.

### Data Augmentation 

In order to improve the performance of the model, **Zoom in** was used. In **Tensorflow** the *tf.image* module contains various functions for image processing and image augmentation. 

For Zoom in the patch, we used *central_crop* :
```javascript
def zoomin(image, mask, img_tab, mask_tab):
        img = np.zeros((64,64,1))
        img[:,:,0] = image
        patch_img_aug = cv2.resize(np.float32(tf.image.central_crop(img, central_fraction=0.5)),(64,64))
        img_tab.append(patch_img_aug)
        msk = np.zeros((64,64,1))
        msk[:,:,0] = mask
        patch_msk_aug = cv2.resize(np.float32(tf.image.central_crop(msk, central_fraction=0.5)),(64,64))
        mask_tab.append(patch_msk_aug)
        return 1
```
Some examples of the use of this function :

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/zoomin.png)

This data augmentation will improve the model ability to predict the segmentation on "weird" MRI, where the size of the folds is not the same as in humans.
By zooming into the center of the patch, we try to generalize the network.
### Training

To train the model, **Adam** was used as an optimizer and **binary crossentropy** as loss function.

To evaluate the model we used two of the most common metrics for semantic segmentation :

* The Intersection-Over-Union (IoU)
* The Dice Coefficient (F1 Score)

### Results

Some examples of the results of the network : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/2D-patch3.png)

Let's focus on one 2D-Patch :

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/2D-patch-pred.png)

For this example the dice score and the IoU were calculated for each tissue : 

---| Dice Score| IoU 
--- | --- | ---
WM | 0.9717| - 
GM | 0.9731| - 
CSF | 0.9717| -

Let's have a look at the segmentation of the reconstructed mask : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/2Dpatchreconstructed_x.png)

The dice score is 0.9402 and the IoU is 0.8871

## Third Approach : 2D Patches Segmentation with a ResUNet Architecture

### Architecture

The idea of this architecture is to replace the convolutions in U-Net on each level with ResBlocks in order to improve the performance of our previous model. The residual blocks with skip connections helped in making a deeper and deeper convolution neural network.

The general architecture of the Unet with ResBlocks is detailled below: 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/U-Resnet_architecture.png)

To compare with the previous network, we have to implement the **ResBlock** function. It's composed of a **shortcut** which will connect the output of one layer with the input of an earlier layer. 

``` python
def ResBlock(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path
  ```  
   
### Results

Some examples of the results of the network : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/RESUNET_RES.PNG)

Here we had a dice score of 0.9476 and a IoU of ....
Thus, the results are better with this kind of architecture. 

Let's focus on one 2D-Patch :

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/res_resunet.PNG)

For this example the dice score and the IoU were calculated for each tissue : 

---| Dice Score| IoU 
--- | --- | ---
WM | 0.9566| - 
GM | 0.9801| - 
CSF | 0.9354| -


## Sources 
*U-Net: Convolutional Networks for Biomedical Image Segmentation, Olaf Ronneberger, Philipp Fischer, Thomas Brox https://arxiv.org/abs/1505.04597
###
