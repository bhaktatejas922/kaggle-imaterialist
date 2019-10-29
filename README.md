# Steel Defect Detection using Hybrid Task Cascade Model.

What is Hybrid Task Cascade?

HTC, or Hybrid Task Cascade is a cascade of multiple top performing models. It is a very complex model with thousands of layers that if training from scratch, takes weeks to train on even the best setups.

Cascade is a classic yet powerful architecture that has
boosted performance on various tasks. However, how to introduce cascade to instance segmentation remains an open
question. A simple combination of Cascade **R-CNN** and
**Mask R-CNN** only brings limited gain. In exploring a more
effective approach, we find that the key to a successful instance segmentation cascade is to fully leverage the reciprocal relationship between detection and segmentation. In
this work, we propose a new framework, Hybrid Task Cascade (HTC), which differs in two important aspects: (1) instead of performing cascaded refinement on these two tasks
separately, it interweaves them for a joint multi-stage processing; (2) it adopts a fully convolutional branch to provide spatial context, which can help distinguishing hard
foreground from cluttered background. Overall, this framework can learn more discriminative features progressively
while integrating complementary features together in each
stage. Without bells and whistles, a single HTC obtains
38.4% and 1.5% improvement over a strong Cascade Mask
R-CNN baseline on MSCOCO dataset. Moreover, our overall system achieves 48.6 mask AP on the test-challenge split,
**ranking 1st in the COCO 2018 Challenge Object Detection**
Task. Code is available at: https://github.com/
open-mmlab/mmdetection.

![ensemble](figures/prediction.png)

## Solution
My solution is based on the COCO challenge 2018 winners article: https://arxiv.org/abs/1901.07518. 

### Model: 
[Hybrid Task Cascade with ResNeXt-101-64x4d-FPN backbone](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py). This model has a metric Mask mAP = 43.9 on COCO dataset. This is SOTA for instance segmentation.

### Validation:
For validation, I used 450 training samples splitted using https://github.com/trent-b/iterative-stratification.

### Preprocessing:
I applied light augmentatios from the [albumentations](https://github.com/albu/albumentations) library to the original image. Then I use multi-scale training: in each iteration, the scale of short edge is randomly sampled
from [600, 1200], and the scale of long edge is fixed as 1900.

![preprocessing](figures/preproc.png)

### Training details:
* pre-train from COCO
* optimizer: `SGD(lr=0.03, momentum=0.9, weight_decay=0.0001)`
* batch_size: 2 = 2 images, Tesla V100
* learning rate scheduler:
```
if iterations < 500:
   lr = warmup(warmup_ratio=1 / 3)
if epochs == 10:
   lr = lr ∗ 0.1
if epochs == 18:
   lr = lr ∗ 0.1
if epochs > 20:
   stop
```
* training time: ~3 days.

To see code for setup, pre-processing, training, and testing, please see Jupyter Notebook here: https://github.com/bhaktatejas922/kaggle-imaterialist/blob/master/mmdetection.ipynb



## References
* https://github.com/open-mmlab/mmdetection

* The First Place Solution of [iMaterialist (Fashion) 2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/)
