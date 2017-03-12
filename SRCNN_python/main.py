from model import SRCNN
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 6, "Number of epoch.")
tf.app.flags.DEFINE_integer('batch_size', 128,"Number of samples per batch.")
tf.app.flags.DEFINE_integer('image_size', 33,"The size of image to use.")
tf.app.flags.DEFINE_integer("label_size", 21, "The size of label to produce.")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm.")
tf.app.flags.DEFINE_integer("input_channel", 1, "Dimension of image color.")
tf.app.flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image.")
tf.app.flags.DEFINE_string("checkpoint_dir", "D:/workspace/3rdPorject_SRCNN/SRCNN/checkpoint", "Name of checkpoint directory.")
tf.app.flags.DEFINE_string("sample_dir", "D:/workspace/3rdPorject_SRCNN/SRCNN/sample", "Name of sample directory.")
tf.app.flags.DEFINE_string("summaries_dir", "D:/workspace/3rdPorject_SRCNN/SRCNN/summary", "Name of sample directory.")
tf.app.flags.DEFINE_string("mode", "train", "train or test")
tf.app.flags.DEFINE_boolean("is_train", True, "True for training, False for testing.")

def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
    sess = tf.InteractiveSession()

    
    srcnn=SRCNN(sess, 
                  imageSize=FLAGS.image_size, 
                  labelSize=FLAGS.label_size, 
                  batchSize=FLAGS.batch_size,
                  channel=FLAGS.input_channel, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)
        
    srcnn.mode()
if __name__ == '__main__':
    tf.app.run()




