# tensorflow-vdsr

## Overview
This is a Tensorflow implementation for ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 16'](http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf).
- [The author's project page](http://cv.snu.ac.kr/research/VDSR/)
- To download the required data for training/testing, please refer to the README.md at data directory.

## Files
- VDSR.py : main training file.
- MODEL.py : model definition.
- MODEL_FACTORIZED.py : model definition for Factorized CNN. (not recommended to use. for record purpose only)
- PSNR.py : define how to calculate PSNR in python
- TEST.py : test all the saved checkpoints
- PLOT.py : plot the test result from TEST.py

## How To Use
### Training
```shell
# if start from scratch
python VDSR.py
# if start with a checkpoint
python VDSR.py --model_path ./checkpoints/CHECKPOINT_NAME.ckpt
```
### Testing
```shell
# this will test all the checkpoint in ./checkpoint directory.
# and save the results in ./psnr directory
python TEST.py
```
### Plot Result
```shell
# plot the psnr result stored in ./psnr directory
python PLOT.py
```

## Result
The checkpoint is file is [here](https://drive.google.com/file/d/0B4KsMpU0BeosQS1tMWpUZmlBM1E/view?usp=sharing)


