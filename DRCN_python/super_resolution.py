# coding=utf8

"""
Deeply-Recursive Convolutional Network for Image Super-Resolution
Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html

Test implementation model
Author: Jin Yamanaka
"""

from __future__ import division
import time
import random
import os

import tensorflow as tf
import numpy as np
import super_resolution_utilty as util


class DataSet:

  def __init__(self, cache_dir, filenames, channels=1, scale=1, alignment=0, jpeg_mode=False, max_value=255.0):
    
    self.count = len(filenames)
    self.image = self.count * [None]

    for i in range(self.count):
      image = util.load_input_image_with_cache(cache_dir, filenames[i], channels=channels,
                                               scale=scale, alignment=alignment, jpeg_mode=jpeg_mode, max_value=max_value)
      self.image[i] = image

  def convert_to_batch_images(self, window_size, stride, max_value = 255.0):
    
    batch_images = self.count * [None]
    batch_images_count = 0

    for i in range(self.count):
      image = self.image[i]
      if max_value != 255.0:
        image = np.multiply(self.image[i], max_value / 255.0)
      batch_images[i] = util.get_split_images(image, window_size, stride=stride)
      batch_images_count += batch_images[i].shape[0]

    images = batch_images_count * [None]
    no = 0
    for i in range(self.count):
      for j in range(batch_images[i].shape[0]):
        images[no] = batch_images[i][j]
        no += 1
    
    self.image = images
    self.count = batch_images_count
    
    print ("%d mini-batch images are built." % len(self.image))
    

class DataSets:

  def __init__(self, cache_dir, filenames, scale, batch_size, stride_size, width=0, height=0, channels=1, jpeg_mode=False, max_value=255.0):
    
    self.input = DataSet(cache_dir, filenames, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode)
    self.input.convert_to_batch_images(batch_size, stride_size, max_value=max_value)

    self.true = DataSet(cache_dir, filenames, channels=channels, alignment=scale, jpeg_mode=jpeg_mode)
    self.true.convert_to_batch_images(batch_size, stride_size, max_value=max_value)

class SuperResolution:

  def __init__(self, flags, model_name="model"):
    
    # Model Parameters
    self.lr = flags.initial_lr
    self.lr_decay = flags.lr_decay
    self.lr_decay_epoch = flags.lr_decay_epoch
    self.beta1 = flags.beta1
    self.beta2 = flags.beta2
    self.momentum = flags.momentum
    self.feature_num = flags.feature_num
    self.cnn_size = flags.cnn_size
    self.cnn_stride = 1
    self.inference_depth = flags.inference_depth
    self.batch_num = flags.batch_num
    self.batch_size = flags.batch_size
    self.stride_size = flags.stride_size
    self.optimizer = flags.optimizer
    self.loss_alpha = flags.loss_alpha
    self.loss_alpha_decay = flags.loss_alpha / flags.loss_alpha_zero_epoch
    self.loss_beta = flags.loss_beta
    self.weight_dev = flags.weight_dev
    self.initializer = flags.initializer

    # Image Processing Parameters
    self.scale = flags.scale
    self.max_value = flags.max_value
    self.channels = flags.channels
    self.jpeg_mode = flags.jpeg_mode

    # Training or Other Parameters
    self.checkpoint_dir = flags.checkpoint_dir
    self.model_name = model_name

    # Debugging or Logging Parameters
    self.log_dir = flags.log_dir
    self.debug = flags.debug
    self.visualize = flags.visualize
    self.summary = flags.summary
    self.log_weight_image_num = 16

    # initializing variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    self.sess = tf.InteractiveSession(config=config)
    self.H_conv = (self.inference_depth + 1) * [None]
    self.batch_input_images = self.batch_num * [None]
    self.batch_true_images = self.batch_num * [None]

    self.index_in_epoch = -1
    self.epochs_completed = 0
    self.min_validation_mse = -1
    self.min_validation_epoch = -1

    util.make_dir(self.log_dir)
    util.make_dir(self.checkpoint_dir)
    if flags.initialise_log:
      util.clean_dir(self.log_dir)

    print ("Features:%d Inference Depth:%d Initial LR:%0.5f [%s]" % \
          (self.feature_num, self.inference_depth, self.lr, self.model_name))

  def load_datasets(self, cache_dir, training_filenames, test_filenames, batch_size, stride_size):
    self.train = DataSets(cache_dir, training_filenames, self.scale, batch_size, stride_size,
                          channels=self.channels, jpeg_mode=self.jpeg_mode, max_value=self.max_value)
    self.test = DataSets(cache_dir, test_filenames, self.scale, batch_size, batch_size,
                          channels=self.channels, jpeg_mode=self.jpeg_mode, max_value=self.max_value)

  def set_next_epoch(self):

    self.loss_alpha = max(0, self.loss_alpha - self.loss_alpha_decay)

    self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
    self.epochs_completed += 1
    self.index_in_epoch = 0
    
  def build_training_batch(self):

    self.trained = 0
    self.training_psnr =0

    if self.index_in_epoch < 0:
      self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
      self.index_in_epoch = 0
    
    for i in range(self.batch_num):
      if self.index_in_epoch >= self.train.input.count:
        self.set_next_epoch()
        
      self.batch_input_images[i] = self.train.input.image[self.batch_index[self.index_in_epoch] ]
      self.batch_true_images[i] = self.train.true.image[self.batch_index[self.index_in_epoch] ]
      self.index_in_epoch += 1

  def build_embedding_graph(self):
    
    self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="X")
    self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="Y")

    # H-1 conv
    self.Wm1_conv = util.weight([self.cnn_size, self.cnn_size, self.channels, self.feature_num],
                                               stddev=self.weight_dev, name="W-1_conv", initializer=self.initializer)
    self.Bm1_conv = util.bias([self.feature_num], name="B-1")
    Hm1_conv = util.conv2d_with_bias_and_relu(self.x, self.Wm1_conv, self.cnn_stride, self.Bm1_conv, name="H-1")

    # H0 conv
    self.W0_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.feature_num],
                                             stddev=self.weight_dev, name="W0_conv", initializer=self.initializer)
    self.B0_conv = util.bias([self.feature_num], name="B0")
    self.H_conv[0] = util.conv2d_with_bias_and_relu(Hm1_conv, self.W0_conv, self.cnn_stride, self.B0_conv, name="H0")

    if self.summary:
      # convert to tf.summary.image format [batch_num, height, width, channels]
      Wm1_transposed = tf.transpose(self.Wm1_conv, [3, 0, 1, 2])
      tf.summary.image("W-1" + self.model_name, Wm1_transposed, max_outputs=self.log_weight_image_num)
      util.add_summaries("B-1:" + self.model_name, self.Bm1_conv, mean=True, max=True, min=True)
      util.add_summaries("W-1:" + self.model_name, self.Wm1_conv, mean=True, max=True, min=True)

      util.add_summaries("B0:" + self.model_name, self.B0_conv, mean=True, max=True, min=True)
      util.add_summaries("W0:" + self.model_name, self.W0_conv, mean=True, max=True, min=True)

  def build_inference_graph(self):
    
    if self.inference_depth <= 0:
      return

#    self.W_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.feature_num], stddev=self.weight_dev, name="W_conv", initializer="diagonal")
    self.W_conv = util.diagonal_weight([self.cnn_size, self.cnn_size, self.feature_num, self.feature_num], name="W_conv")
    self.B_conv = util.bias([self.feature_num], name="B")

    for i in range(0, self.inference_depth):
      self.H_conv[i+1] = util.conv2d_with_bias_and_relu(self.H_conv[i], self.W_conv, 1, self.B_conv, name="H%d"%(i+1))

    if self.summary:
      util.add_summaries("W:" + self.model_name, self.W_conv, mean=True, max=True, min=True)
      util.add_summaries("B:" + self.model_name, self.B_conv, mean=True, max=True, min=True)

  def build_reconstruction_graph(self):
    
    # HD+1 conv
    self.WD1_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.feature_num],
                                             stddev=self.weight_dev, name="WD1_conv", initializer=self.initializer)
    self.BD1_conv = util.bias([self.feature_num], name="BD1")

    # HD+2 conv
    self.WD2_conv = util.weight([self.cnn_size, self.cnn_size, self.feature_num, self.channels],
                                              stddev=self.weight_dev, name="WD2_conv", initializer=self.initializer)
    self.BD2_conv = util.bias([1], name="BD2")

    self.Y1_conv = (self.inference_depth + 1) * [None]
    self.Y2_conv = (self.inference_depth + 1) * [None]
    self.W = tf.Variable( np.full(fill_value=1.0 / (self.inference_depth + 1), shape=[self.inference_depth + 1], dtype=np.float32), name="layer_weight")
    W_sum = tf.reduce_sum(self.W)

    for i in range(0, self.inference_depth+1):
      self.Y1_conv[i] = util.conv2d_with_bias_and_relu(self.H_conv[i], self.WD1_conv, self.cnn_stride, self.BD1_conv, name="Y%d_1"%i)
      self.Y2_conv[i] = util.conv2d_with_bias_and_relu(self.Y1_conv[i], self.WD2_conv, self.cnn_stride, self.BD2_conv, name="Y%d_2"%i)
      y_ = tf.mul(self.W[i], self.Y2_conv[i], name="Y%d_mul" % i)
      y_ = tf.div(y_, W_sum, name="Y%d_div" % i)
      if i == 0:
        self.y_ = y_
      else:#skip-connection for layer greater than 0
        self.y_ = self.y_ + y_

    if self.summary:
      util.add_summaries("BD1:" + self.model_name, self.BD1_conv)
      util.add_summaries("WD1:" + self.model_name, self.WD1_conv, mean=True, max=True, min=True)
      util.add_summaries("WD2:" + self.model_name, self.WD2_conv, mean=True, max=True, min=True)


  def build_optimizer(self):
    
    self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
    self.loss_alpha_input = tf.placeholder(tf.float32, shape=[], name="Alpha")

    mse = tf.reduce_mean(tf.square(self.y_ - self.y), name="Loss1")
    if self.debug:
      mse = tf.Print(mse, [mse], message="MSE: ")

    if self.loss_alpha == 0.0 or self.inference_depth == 0:
      loss = mse
    else:
      loss1_mse = self.inference_depth * [None]

      for i in range(0, self.inference_depth):
        inference_sub = tf.sub(self.y, self.Y2_conv[i], name="Loss1_%d_sub" % i)
        inference_square = tf.square(inference_sub, name="Loss1_%d_squ" % i)
        loss1_mse[i] = tf.reduce_mean(inference_square, name="Loss1_%d" % i)

      loss1 = loss1_mse[0]
      for i in range(1, self.inference_depth):
        if i == self.inference_depth:
          loss1 = tf.add(loss1, loss1_mse[i], name="Loss1")
        else:
          loss1 = tf.add(loss1, loss1_mse[i], name="Loss1_%d_add" % i)

      loss1 = tf.mul(1.0 / self.inference_depth, loss1, name="Loss1_weight")
      loss2 = mse
      if self.visualize:
        tf.summary.scalar("L1:" + self.model_name, loss1)
        tf.summary.scalar("L2:" + self.model_name, loss2)
      loss1 = tf.mul(self.loss_alpha_input, loss1, name="Loss1_alpha")
      loss2 = tf.mul(1 - self.loss_alpha_input, loss2, name="Loss2_alpha")

      if self.loss_beta > 0.0:
        with tf.name_scope('Loss3') as scope:
          loss3 = tf.nn.l2_loss(self.Wm1_conv) + tf.nn.l2_loss(self.W0_conv) \
                  + tf.nn.l2_loss(self.W_conv) + tf.nn.l2_loss(self.WD1_conv) \
                  + tf.nn.l2_loss(self.WD2_conv)
          loss3 *= self.loss_beta

        if self.visualize:
          tf.summary.scalar("L3:" + self.model_name, loss3)
        loss = loss1 + loss2 + loss3
      else:
        loss = loss1 + loss2

    if self.visualize:
      tf.summary.scalar("Loss:" + self.model_name, loss)

    self.loss = loss
    self.mse = mse
    self.train_step = self.add_optimizer_op(loss, self.lr_input)

  def add_optimizer_op(self, loss, lr_input):
    
    if self.optimizer == "gd":
      train_step = tf.train.GradientDescentOptimizer(lr_input).minimize(loss)
    elif self.optimizer == "adadelta":
      train_step = tf.train.AdadeltaOptimizer(lr_input).minimize(loss)
    elif self.optimizer == "adagrad":
      train_step = tf.train.AdagradOptimizer(lr_input).minimize(loss)
    elif self.optimizer == "adam":
      train_step = tf.train.AdamOptimizer(lr_input, beta1=self.beta1, beta2=self.beta2).minimize(loss)
    elif self.optimizer == "momentum":
      train_step = tf.train.MomentumOptimizer(lr_input, self.momentum).minimize(loss)
    elif self.optimizer == "rmsprop":
      train_step = tf.train.RMSPropOptimizer(lr_input, momentum=self.momentum).minimize(loss)
    else:
      print ("Optimizer arg should be one of [gd, adagrad, adam, momentum, rmsprop].")
      return None

    return train_step

  def init_all_variables(self, load_initial_data=False):
    
    if self.visualize:
      self.summary_op = tf.summary.merge_all()
      self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

    if load_initial_data:
      self.saver.restore(self.sess, self.checkpoint_dir + "/" + self.model_name + ".ckpt")
      print("Model restored.")

    self.start_time = time.time()

  def train_batch(self, step=0, log_mse=False):

    _, mse = self.sess.run([self.train_step, self.mse], feed_dict={self.x: self.batch_input_images,
                                              self.y: self.batch_true_images,
                                              self.lr_input: self.lr,
                                              self.loss_alpha_input: self.loss_alpha})
    self.trained += 1
    self.training_psnr += util.get_psnr(mse, max_value=self.max_value)


  def evaluate(self, step):
    
    summary_str, mse = self.sess.run([self.summary_op, self.mse],
                                    feed_dict={self.x: self.test.input.image,
                                              self.y: self.test.true.image,
                                              self.loss_alpha_input: self.loss_alpha})

    self.summary_writer.add_summary(summary_str, step)
    self.summary_writer.flush()

    if self.min_validation_mse < 0 or self.min_validation_mse > mse:
      self.min_validation_epoch = self.epochs_completed
      self.min_validation_mse = mse
    else:
      if self.epochs_completed > self.min_validation_epoch + self.lr_decay_epoch:
        self.min_validation_epoch = self.epochs_completed
        self.min_validation_mse = mse
        self.lr *= self.lr_decay

    return mse

  def save_summary(self):

    summary_str = self.sess.run(self.summary_op,
                                    feed_dict={self.x: self.test.input.image,
                                              self.y: self.test.true.image,
                                              self.loss_alpha_input: self.loss_alpha})

    self.summary_writer.add_summary(summary_str, 0)
    self.summary_writer.flush()


  def print_status(self, step, mse):
    
    processing_time = (time.time() - self.start_time) / step

    if self.trained > 0:
      training_psnr = self.training_psnr / self.trained
    else:
      training_psnr =0

    print ("%s Step:%d MSE:%f PSNR:%f (%f)" % (util.get_now_date(), step, mse, util.get_psnr(mse, max_value=self.max_value), training_psnr))
    print ("Epoch:%d LR:%f α:%f (%2.2fsec/step)" % (self.epochs_completed, self.lr, self.loss_alpha, processing_time))

  def print_weight_variables(self):
    
    util.print_CNN_weight(self.Wm1_conv)
    util.print_CNN_bias(self.Bm1_conv)
    util.print_CNN_weight(self.W0_conv)
    util.print_CNN_bias(self.B0_conv)
    util.print_CNN_bias(self.W)

  def save(self):
    
    filename = self.checkpoint_dir + "/" + self.model_name + ".ckpt"
    self.saver.save(self.sess, filename)
    print("Model saved [%s]." % filename)

  def do(self, input_image):
  
    if len(input_image.shape) == 2:
      input_image = input_image.reshape(input_image.shape[0], input_image.shape[1], 1)
  
    image = np.multiply(input_image, self.max_value / 255.0)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    y = self.sess.run(self.y_, feed_dict={self.x: image})

    return np.multiply(y[0], 255.0 / self.max_value)
  
  def do_super_resolution(self, file_path, output_folder):
    
    filename, extension = os.path.splitext(file_path)
    org_image = util.load_image(file_path)
    input_image = util.resize_image_by_bicubic(org_image, self.scale)
    util.save_image("output/" + file_path, org_image)
    
    if len(input_image.shape) >= 3 or input_image.shape[2] == 3 and self.channels == 1:
      # use y_image for rgb_image
      input_ycbcr_image = util.convert_rgb_to_ycbcr(input_image, jpeg_mode=self.jpeg_mode)
      input_y_image = input_ycbcr_image[:, :, 0:1].copy()
      output_y_image = self.do(input_y_image)
      image = util.convert_y_and_cbcr_to_rgb(output_y_image, input_ycbcr_image, jpeg_mode=self.jpeg_mode)
    else:
      image = self.do(org_image)
      
    util.save_image("output/" + filename + "_result" + extension, image)
    return 0

  def do_super_resolution_for_test(self, file_path, output_folder):

    filename, extension = os.path.splitext(file_path)
    true_image = util.set_image_alignment(util.load_image(file_path), self.scale)
    util.save_image("output/" + file_path, true_image)

    input_image = util.load_input_image(file_path, channels=self.channels, scale=self.scale, alignment=self.scale,
                             convert_ycbcr=True, jpeg_mode=self.jpeg_mode)
    util.save_image("output/" + filename + "_input" + extension, input_image)
    input_color_image = util.load_input_image(file_path, channels=3, scale=self.scale, alignment=self.scale,
                             convert_ycbcr=False, jpeg_mode=self.jpeg_mode)
    util.save_image("output/" + filename + "_input_c" + extension, input_color_image)

    if len(true_image.shape) >= 3 and true_image.shape[2] == 3 and self.channels == 1:
      true_image = util.convert_rgb_to_y(true_image, jpeg_mode=self.jpeg_mode)
      util.save_image("output/" + filename + "_true" + extension, true_image)
      input_ycbcr_image = util.load_input_image(file_path, channels=3, scale=self.scale, alignment=self.scale,
                                          convert_ycbcr=True, jpeg_mode=self.jpeg_mode)
      output_image = self.do(input_image)
      output_color_image = util.convert_y_and_cbcr_to_rgb(output_image, input_ycbcr_image[:,:,1:3], jpeg_mode=self.jpeg_mode)
      util.save_image("output/" + filename + "_result_c" + extension, output_color_image)
    else:
      # for monochro or rgb image
      output_image = self.do(input_image)

    mse = util.compute_mse(true_image, output_image, border_size=self.scale)

    util.save_image("output/" + filename + "_result" + extension, output_image)
    print ("MSE:%f PSNR:%f" % (mse, util.get_psnr(mse)))
    return mse

  def end_train_step(self, step):
    total_time = time.time() - self.start_time
    processing_time = total_time / step
    
    h = total_time // (60 * 60)
    m = (total_time - h * 60 * 60) // 60
    s = (total_time - h * 60 * 60 - m * 60)
  
    print ("Finished at Step:%d. Total time:%02d:%02d:%02d (%0.3fsec/step)\n" % (step, h, m, s, processing_time))
