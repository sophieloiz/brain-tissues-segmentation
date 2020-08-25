# 2D Patches Segmentation

Neuroscientists who want to work on non-human MRI can't use standard segmentation tools like **FAST FSL** to segmentate brain tissues like white and grey matter. In order to avoid manual Segmentation which is booth challenging and time consuming, we try to developp a deep learning approach by using small patches. The idea was instead of training the network on complete slices, to train the network on pieces of the image so that we can use it for non-human data.  

## Preprocessing 

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
