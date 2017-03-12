from  utils import *
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image as im
import scipy.misc
import scipy.ndimage

class SRCNN(object):
    
    def __init__(self,sess,imageSize=33,
                 labelSize=21,batchSize=128,channel=1,
                 checkpoint_dir=None,sample_dir=None):
        self.sess=sess
        self.imageSize=imageSize
        self.labelSize=labelSize
        self.batchSize=batchSize
        self.channel=channel
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.setupModel()
        self.saver=tf.train.Saver()
    
    def setupModel(self):
        
        self.images=tf.placeholder(tf.float32, shape=[None,self.imageSize,self.imageSize,self.channel], name='inputs')
        self.labels=tf.placeholder(tf.float32, shape=[None,self.labelSize,self.labelSize,self.channel], name='labels')
        
        self.weights={'w1':tf.Variable(tf.random_normal([9,9,1,64],stddev=1e-3,name='w1')),
                      'w2':tf.Variable(tf.random_normal([1,1,64,32],stddev=1e-3,name='w2')),
					  'wx':tf.Variable(tf.random_normal([3,3,32,32],stddev=1e-3,name='wx')),
                      'w3':tf.Variable(tf.random_normal([5,5,32,1],stddev=1e-3,name='w3'))}
        
        self.biases={'b1':tf.Variable(tf.zeros([1],name='b1')),
                   'b2':tf.Variable(tf.zeros([1],name='b2')),
                   'b3':tf.Variable(tf.zeros([1],name='b3'))}
        
        self.predict=self.inference()
        self.loss=self.getloss()
        
        
    def inference(self):
        
        with tf.name_scope('conv1') as scope:
            conv1=tf.nn.conv2d(self.images,self.weights['w1'], strides=[1,1,1,1], padding='VALID',name='weighs' )+ self.biases['b1']
            conv1=tf.contrib.layers.batch_norm(conv1)
            conv1=tf.nn.relu(conv1)
        
        with tf.name_scope('conv2') as scope:
            conv2=tf.nn.conv2d(conv1,self.weights['w2'], strides=[1,1,1,1], padding='VALID',name='weighs' )+ self.biases['b2']
            #add batch normalization,should be removed when testing
			conv2=tf.contrib.layers.batch_norm(conv2)
            conv2=tf.nn.relu(conv2)
		
        		
		#add more layers to see how result changes
        addlayernumber=10  #5,10,15
        with tf.name_scope('convadd') as scope:
            for i in range(addlayernumber):
                conv2=tf.nn.conv2d(conv2,self.weights['wx'], strides=[1,1,1,1], padding='SAME',name='weighs' )+ self.biases['b2']
                conv2=tf.contrib.layers.batch_norm(conv2)
                conv2=tf.nn.relu(conv2)
        
        with tf.name_scope('conv3') as scope:
            conv3=tf.nn.conv2d(conv2,self.weights['w3'], strides=[1,1,1,1], padding='VALID',name='weighs' )+ self.biases['b3']
            conv3=tf.nn.relu(conv3)    
        
        return conv3
        
    def getloss(self):
        with tf.name_scope('loss'):
		    #probably add exponential decay learning rate
            loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.labels-self.predict), reduction_indices=0))
            tf.scalar_summary('Mean square error', loss)
        return loss
    
    def train(self):
        train_data, train_label = read_data("train.h5")
        train_data=np.transpose(train_data, [0,2,3,1])
        train_label=np.transpose(train_label, [0,2,3,1])
        print(train_data.shape,train_label.shape)
        self.train_op =tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss)
        merged = tf.merge_all_summaries()
        train_writer=tf.train.SummaryWriter(FLAGS.summaries_dir)
        tf.initialize_all_variables().run()
        
        counter = 0
        start_time = time.time()
        print('Start training....')
        for x in range(FLAGS.epoch):
            batchNumber=len(train_data)//self.batchSize
            for batch in range(0,batchNumber):
                batch_images = train_data[batch*FLAGS.batch_size : (batch+1)*FLAGS.batch_size]
                batch_labels = train_label[batch*FLAGS.batch_size : (batch+1)*FLAGS.batch_size]
                counter += 1
                if counter % 10 == 0:
                    run_metadata = tf.RunMetadata()
                    summary,_, err=self.sess.run([merged,self.train_op,self.loss],feed_dict={self.labels:batch_labels,self.images:batch_images})
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % counter)
                    train_writer.add_summary(summary, counter)
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((x+1), counter, time.time()-start_time, err))
                if counter==500:
                    self.save(FLAGS.checkpoint_dir, counter)
        train_writer.close()        
    def test(self):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        test_data, test_label = read_data("test.h5")
        test_data=np.transpose(test_data, [0,2,3,1])
        test_label=np.transpose(test_label, [0,2,3,1])
        print(len(test_data),len(test_label))
        result=self.sess.run([self.predict],feed_dict={self.labels:test_label,self.images:test_data})
        image_path = os.path.join(os.getcwd(), FLAGS.sample_dir)
        
        image_path = os.path.join(image_path, "test_image"+str(1)+".png")
        print (result[0][0,:,:,0])
        scipy.misc.imsave('test.png',result[0][0,:,:,0])
        
    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.labelSize)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    def load(self, checkpoint_dir):
        print(" Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.labelSize)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        #Returns CheckpointState proto from the "checkpoint" file.
        #If the "checkpoint" file contains a valid CheckpointState proto, returns it.
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False   
        
    
    def mode(self):
        if FLAGS.mode=="train":
            self.train()
        else:
            self.test()
        