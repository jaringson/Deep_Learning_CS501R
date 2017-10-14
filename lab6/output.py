import numpy as np 
import tensorflow as tf
import PIL as Image

#tf.reset_default_graph()

sess = tf.InteractiveSession()



batch_size = 25

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, transpose=False, name="conv" ):
    
    with tf.name_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        #W = weight_variable([filter_size, filter_size, x.get_shape().as_list()[3], num_filters])
        W = tf.Variable( 1e-3*np.random.randn(filter_size, filter_size, x_shape[3], num_filters).astype(np.float32))
        b = bias_variable([num_filters])
        h = []
        

        if not is_output:
            if  transpose:
                h = tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=[batch_size ,x_shape[1]*2 ,x_shape[2]*2 ,x_shape[3]], strides=[1,stride,stride,1], padding='SAME') + b)
            else:
                h = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b)
        else:
            h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b
        return h


def fc( x, out_size=50, is_output=False, name="fc" ):
    with tf.name_scope(name) as scope:
        #W = weight_variable( [x.get_shape().as_list()[1], out_size])
        W = tf.Variable( 1e-3*np.random.randn(x.get_shape().as_list()[1], out_size).astype(np.float32))
        b = bias_variable([out_size])
        h = []
        if not is_output:
            h = tf.nn.relu(tf.matmul(x,W)+b)
        else:
            h = tf.matmul(x,W)+b
        return h

x_image = tf.placeholder(tf.float32, shape=[batch_size, 512/2,512/2,3], name="images")
label_ = tf.placeholder(tf.float32, shape=[batch_size, 512/2,512/2,2])
keep_prob = tf.placeholder(tf.float32)

#x_image = tf.reshape(x, [-1,512,512,3])



saver = tf.train.Saver()
saver.restore(sess, "script/tf_logs/lab6.ckpt")
print("Whole model restored")

img = np.array(Image.open('./cancer_data/input/pos_test_000072.png'))
print img.shape
img = np.tile(img,(25,1)) 
print img.shape

lb = sess.run([label_conv],feed_dict={x_image:img, keep_prob:1.0})

print lb.shape
