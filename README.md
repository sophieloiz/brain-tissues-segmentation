# Brain Tissues Segmentation

## Introduction 
The purpose of this project is to develop deep learning approaches for the segmentation of brain tissues. These segmentations are useful for measuring and visualizing anatomical structures but also to analyze brain changes in case of disease like Alzheimer. Today different automatic segmentation exists like FAST (FSL), Freesurfer or ANTS. But these approaches are often inaccurate and require manual segmentation which is booth time consuming and challenging. 

## First Approach : UNet Implementation

### Preprocessing

The first UNet implemented took 2D images as input. So, we need to slice the 3D Volume images of our dataset.  
This is an example of a single slice of the input image and the associated mask : 

![Image Mask](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/preprocess.png)


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

The U-net is a convolutional network architecture used for fast and precise segmentation of images. This is a very popular architeture in bio medical images. 

The architecture contains two paths:
* a contraction path => encoder
* an expanding path => decoder
 
 The **encoder** is used to capture the context in the image whereas the **decoder** will enable precise localization. 


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


### Results

Some example of the results of the network : 

* For the gray matter segmentation : 

![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/results_pve1.png)

* For the white matter segmentation : 

* For the cerebrospinal fluid segmentation : 


 ---- | Dice Score| IoU
------------ | -------------| -------------
WM | -------------| -------------
GM |  -------------| -------------
CSF |  -------------| -------------
