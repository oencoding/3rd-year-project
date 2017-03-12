# deeply-recursive-cnn-tf

## overview
This project is a test implementation of ["Deeply-Recursive Convolutional Network for Image Super-Resolution", CVPR2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.pdf) using tensorflow


Paper: ["Deeply-Recursive Convolutional Network for Image Super-Resolution"] (https://arxiv.org/abs/1511.04491) by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea


## how to use

```
# train with default parameters and evaluate after training for Set5 (takes whole day to train with high-GPU)
python main.py

# training with simple model (will be good without GPU)
python main.py —-end_lr 1e-4 —-feature_num 32 -—inference_depth 5

# evaluation for set5 only (after training has done)
python main.py -—dataset set5 --is_training False

```

Network graphs and weights / loss summaries are saved in **tf_log** directory.

Weights are saved in **model** directory.


