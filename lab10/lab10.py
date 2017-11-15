
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize

sess = tf.Session()


opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'content.png', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

# print content_acts
# print vgg

content_layer = content_acts[8]
print content_layer.shape

orig_content = ops[8]

print orig_content.get_shape()
# content_loss = tf.reshape(content_loss, [1, 28*28, 512])
# print content_loss.get_shape()

# content_loss = tf.reshape(content_loss, [28*28, 512])
# print content_loss.get_shape()

# content_l = tf.reduce_sum(tf.multiply(content_loss, content_loss))

# print content_l.get_shape()

# print content_acts

# print style_acts

content_loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(content_layer, orig_content)))
#
# --- construct your cost function here
#

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)
style_pos = [0,2,4,7,10]
style_loss = 0
for pos in style_pos:
	style_layer = style_acts[pos]
	orig_style = ops[pos]
	style_layer = tf.constant(style_layer)
	shape = style_layer.get_shape().as_list()
	gram = tf.reshape(style_layer, [1,shape[1]*shape[2], shape[3]])
	gram = tf.reshape(gram, [shape[1]*shape[2], shape[3]])
	gram = tf.reduce_sum(tf.multiply(gram, gram))
	style_loss += (1.0/ (4 * shape[3]**2 * (shape[1]*shape[2])**2 ) ) * tf.reduce_sum(tf.square(tf.subtract(gram, style_img)))
style_loss *= 1.0/5.0

alpha = 1.0
beta = 1e-3

loss = alpha * content_loss + beta * style_loss

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
adam = tf.train.AdamOptimizer(0.1).minimize(loss, var_list=[opt_img])

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

# --- place your optimization loop here
for i in range(100):
	optimze, r_img, l, cl, sl = sess.run([adam, opt_img, loss, content_loss, style_loss])
	print i, l, sl, cl
