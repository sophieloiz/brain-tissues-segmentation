# Brain Tissues Segmentation

## Introduction 
The purpose of this project is to develop deep learning approaches for the segmentation of brain tissues. These segmentations are useful for measuring and visualizing anatomical structures, but also to analyze brain changes in case of diseases like Alzheimer. Today different automatic segmentations are available thanks to FAST (FSL), Freesurfer and ANTS. But these approaches are often inaccurate and require additional manual segmentations which are both time consuming and challenging. 

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

* For the gray matter segmentation : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results_pve1.png)

* For the white matter segmentation : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results_WM.png)

* For the cerebrospinal fluid segmentation : 
![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/first_result_2.png)


 ---- | Dice Score| IoU| Accuracy
------------ | -------------| -------------| -------------
WM | -------------| -------------| -------------
GM |  0.93218| 0.87599 | 0.9566
CSF |  -------------| -------------| -------------

## Second Approach : Unet with ResBlock

### Architecture

The idea of this architecture is to replace the convolutions in U-Net on each level with ResBlock in order to improve the performance of our previous model. 


![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/U-Resnet_architecture.png)


###
