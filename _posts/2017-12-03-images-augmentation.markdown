---
layout: post
title:  "images augmentation"
date:   2017-12-03 20:21:08 +0800
categories: image
---
**This post is the first part of images detection and segmentation.**
# data preparing
 You can find the `water lily` images on [Google][google] or [Baidu][baidu] image, also you can take the photo about water lily by yourself. Here are some examples collected by me:

<img align="left" width="300" height="290" src="/assets/image/blue_1.jpg">
<img align="center" width="300" height="290" src="/assets/image/blue_4.jpg">
<img align="left" width="300" height="290" src="/assets/image/purple_1.jpg">
<img align="center" width="300" height="290" src="/assets/image/red_1.jpeg">

I have collected 20 pictures for each kinds of water lily, the first 15 for train and the others for test. The train set is inadequate so images augmentation should be implemented.

# images augmentation

To augment the images, we will use the [imgaug][imgaug-link]. *It converts a set of input images into a new, much larger set of slightly altered images.* First of all, you should install the package correctly [according this Documentation][imgaug-installation-documentation] . Imgaug include many augmentation techniques, such as Crop, Pad, GaussianBlur. Here is the [quick example code to use the library][quick-use-imgaug].

The following code implement augmentation for one image.

```python
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

data_set_path = './lotus'

temp_blue = io.imread(data_set_path+'/blue_butterfly/blue_1.jpg')

# The array has shape (images_num, width, height, channel) and dtype uint8.
images = np.array(
    [temp_blue for _ in range(10)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-4, 4)
    )
], random_order=True) # apply augmenters in random order

images_aug = seq.augment_images(images)
```
As you can see, the `images` has 4 dimensions shape *[images_num, width, height, channel]*. After augmentation, `images_aug` also has 4 dimensions, and the first dimension is the number of pictures. The follow pictures show one of the augmentation result.

<img align="left" width="300" height="290" src="/assets/image/blue_1.jpg">
<img align="center" width="370" height="325" src="/assets/image/blue_1_aug.png">
<img align="left" width="300" height="290" src="/assets/image/purple_1.jpg">
<img align="center" width="370" height="325" src="/assets/image/purple_1_aug.png">
<img align="left" width="300" height="290" src="/assets/image/red_1.jpeg">
<img align="center" width="370" height="325" src="/assets/image/red_1_aug.png">

My work directory appear as follow:

    - imag_augmentation.py
    + lotus/
      + blue_butterfly/
        - blue_1.jpg
        - blue_2.jpg
        ...
        - blue_20.jpg
      + purple_jade/
        - purple_1.jpg
        - purple_2.jpg
        ...
        - purple_20.jpg
      + red_beauty
        - red_1.jpg
        - red_2.jpg
          ...
        - red_20.jpg

The following code augment the training sets, each image become 11(1+10) images.

```python
import numpy as np
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

blue_path = './lotus/blue_butterfly/'
red_path = './lotus/red_beauty/'
purple_path = './lotus/purple_jade/'

#temp_blue = io.imread(blue_path+np.str(1)+'.jpg')

# The array has shape (images_num, width, height, channel) and dtype uint8.
#images = np.array(
#    [temp_blue for _ in range(10)],
#    dtype=np.uint8
#)

seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-4, 4)
    )
], random_order=True) # apply augmenters in random order

def augment_fun(seq_,picture_path,color,train_num,augment_num):
    for i in range(train_num):
        temp_pic = io.imread(picture_path+color+'_'+np.str(i+1)+'.jpg')
        images = np.array(
        [temp_pic for _ in range(augment_num)],
        dtype=np.uint8
        )

        images_aug = seq.augment_images(images)
        for j in range(augment_num):
            plt.figure()                     # to save the aumentation images
            plt.imshow(images_aug[j])
            plt.savefig(picture_path+color+'_'+np.str(i+1)+np.str(j+1)+'.png')
            temp_png = Image.open(picture_path+color+'_'+np.str(i+1)+np.str(j+1)+'.png')
            temp_png = temp_png.convert('RGB')
            temp_png.save(picture_path+color+'_'+np.str(i+1)+np.str(j+1)+'.jpg')
            plt.close()

augment_fun(seq,blue_path,'blue',15,10)
augment_fun(seq,purple_path,'purple',15,10)
augment_fun(seq,red_path,'red',15,10)
```


[google]: https://www.google.com
[baidu]: https://www.baidu.com
[imgaug-link]: https://github.com/aleju/imgaug
[quick-use-imgaug]:http://imgaug.readthedocs.io/en/latest/source/examples_basics.html
[imgaug-installation-documentation]:http://imgaug.readthedocs.io/en/latest/source/installation.html
