---
layout: post
title:  "Before Retrain"
date:   2018-01-03 22:46:08 +0800
categories: image
---
**In this post I will show you the detection result about Water lily without retrain**

# Tensorflow Object Detection
Last year (2017*), Google open a [tensorflow object detection api][tensorflow object detection], and there is a [tutorial][tensorflow object detection tutorial], which is easy to follow step by step. In the [model zoo][model zoo], they provide models pre-trained on the COCO dataset, the Kitti dataset, and the Open Images dataset. Also, there are several model combinations, like `faster-rcnn-inception-v2...`, `faster-rcnn-resnet...`, `ssd-inception-v2...`, `ssd-mobilenet-v2...` and so on. I will show the result of water lily detection use different pre-trained models.

# Experiment and Result
In this section, I will show the detection result by different models and datasets.

### ssd + mobilenet + coco
In the [model zoo][model zoo], this combination is the faster one, but it's performance (mAP) is almost the worst one. Without retrain, it's detection result shows as following:

<img align="center" width="600" height="430" src="/assets/image/before_train/ssd/image1.png">
<img align="center" width="600" height="450" src="/assets/image/before_train/ssd/image2.png">
<img align="center" width="600" height="400" src="/assets/image/before_train/ssd/image3.png">

As you can see, the result is pretty bad, only the middle one is detected with leaf (and the leaf is recognized as `cake`). The first and third one with nothing be detected.

I also record the model execute time for detecting these three pictures, the result shows as follow:

2.918037891387939453e+00 s

8.569598197937011719e-02 s

9.798288345336914062e-02 s

The time to detect first picture is much longer than the others which contains the preparing time. You will see the similar result in the next several combinations.

### faster_rcnn + resnet50 + coco

Faster_Rcnn is one of my favorite model in `Computer Vision`, and I will try to explain the details about this model in the future post.

<img align="center" width="600" height="430" src="/assets/image/before_train/faster_resnet_coco/image1.png">
<img align="center" width="600" height="450" src="/assets/image/before_train/faster_resnet_coco/image2.png">
<img align="center" width="600" height="400" src="/assets/image/before_train/faster_resnet_coco/image3.png">

This model detect more things than ssd model, but still not that good (the leaf of `Water Lily` is still detected as `cake`. Yep, it looks like cake according to the shape). And the time is:

1.438657617568969727e+01 s

6.586786985397338867e+00 s

6.537290096282958984e+00 s

The result shows that the time of faster_rcnn is much longer than the ssd counterpart.  

### faster_rcnn + resnet101 + coco

<img align="center" width="600" height="430" src="/assets/image/before_train/faster_resnet101_coco/image1.png">
<img align="center" width="600" height="450" src="/assets/image/before_train/faster_resnet101_coco/image2.png">
<img align="center" width="600" height="400" src="/assets/image/before_train/faster_resnet101_coco/image3.png">

To compare with the previous one, this model is just change the layer of ResNet from 50 to 100. It gives very similar results as the previous one. The time of detection is:

1.590241503715515137e+01 s

7.371901988983154297e+00 s

7.448210954666137695e+00 s

### faster_rcnn + resnet101 + kitti

<img align="center" width="600" height="430" src="/assets/image/before_train/kitti/image1.png">
<img align="center" width="600" height="450" src="/assets/image/before_train/kitti/image2.png">
<img align="center" width="600" height="400" src="/assets/image/before_train/kitti/image3.png">

The detection result show that nothing is detected by this model combination, so maybe you shouldn't use this combination without retrain. The execute time show as follow:

1.128367519378662109e+01 s

7.326383113861083984e+00 s

7.521016120910644531e+00 s

### faster_rcnn + inception + resnet + open images  

<img align="center" width="600" height="430" src="/assets/image/before_train/faster_inception_resnet_open/image1.png">
<img align="center" width="600" height="450" src="/assets/image/before_train/faster_inception_resnet_open/image2.png">
<img align="center" width="600" height="400" src="/assets/image/before_train/faster_inception_resnet_open/image3.png">

When you first look at the result, it's pretty good. The detection box just locate around the object. But when you check the objection category, the result is wired. It recognized the water lily as cat. And the time is much longer than the model before:  

5.910059189796447754e+01 s

2.576212596893310547e+01 s

2.450185585021972656e+01 s

The jupyter notebook code is [here][code address] and you can change the model by change the `MODEL_NAME`.

[tensorflow object detection]: https://github.com/tensorflow/models/tree/master/research/object_detection
[tensorflow object detection tutorial]:https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
[model zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[code address]: https://github.com/Newsteinwell/Newsteinwell.github.io/blob/master/assets/code/object_detection_tutorial.ipynb
