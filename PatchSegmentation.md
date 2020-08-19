# 2D Patches Segmentation

Instead of training the network on complete slices, the idea was to train the network on pieces of the image so that we can use it for non-human data.  

## Preprocessing 

The UNet implemented took 2D patches of size : 64x64 as input. Thus, we had to generate several patches per MRI slice. 

A first approach was to resize each slice to (256x256) and then to divide it into 16 patches.


![](https://github.com/sophieloiz/brain-tissues-segmentation/blob/master/img/Input-patch.png)

### Data Augmentation 

## Training

To train the model, **Adam** was used as an optimizer and **binary crossentropy** as loss function.

To evaluate the model we used two of the most common metrics for semantic segmentation :

* The Intersection-Over-Union (IoU)
* The Dice Coefficient (F1 Score)

## Results

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
