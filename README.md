# 3rd-year-project
Here are the code that have been used during the project. Some of the code are modified by myself in order to analyze different models and visualize the result.

The main part of the experiment is carried out in SRCNN where I have tried various structures of the network, for example, adding more layers, increase filter size,etc. Also, other methods such  as batch normalization and residual network is used to compare the result.

VDSR is an improved version of SRCNN which achieved better result and faster training speed as a reuslt of using larger learning rate and residual structure.

DRCN is very similar to VDSR where a recursive structure is used to reduce learning parameters.

DCGAN is the final part of the experiment.This model is a very hot topic in deep learning which is build on the concept of min-max game or so called zero-sum game. It has now been widely used in computer vision related problem and achieves state-of-the-art result. 



#References
1. https://github.com/tegg89/SRCNN-Tensorflow#references
2. https://github.com/liliumao/Tensorflow-srcnn
3. https://github.com/carpedm20/DCGAN-tensorflow
4. https://github.com/david-gpu/srez
5. https://github.com/Jongchan/tensorflow-vdsr
6. https://github.com/jiny2001/deeply-recursive-cnn-tf
7. http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
