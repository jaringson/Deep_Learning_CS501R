import numpy as np 
import tensorflow as tf
import batch_utils
from scipy.misc import imsave
import simplejson

tf.reset_default_graph()
sess = tf.InteractiveSession()

BATCH_SIZE = 100
LAMBDA = 10 # Gradient penalty lambda hyperparameter

g_scope = 0
#d_scope = 0
noise = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def conv_t( x, filter_size=3, stride=1, num_filters=64, is_output=False,out_size=None, name="conv_t"):
    with tf.variable_scope(name) as scope:
    
   
        x_shape = x.get_shape().as_list()
        W = tf.get_variable("W_conv_t", [filter_size, filter_size, num_filters,x_shape[3]], initializer = tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_conv_t", [num_filters], initializer = tf.contrib.layers.variance_scaling_initializer())
        
	if out_size ==None:
            outsize = x_shape
            outsize[0] = BATCH_SIZE
            outsize[1] *= 2
            outsize[2] *= 2
            outsize[3] = W.get_shape().as_list()[2]

        h = []
        

        if not is_output:
            h = tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=outsize, strides=[1, stride, stride, 1], padding="SAME") + b)        
		
        else:
            h = tf.nn.conv2d_transpose(x, W, output_shape=outsize, strides=[1,stride,stride,1], padding='SAME') + b
	

        return h #result

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv"):
    with tf.variable_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        W = tf.get_variable("W_conv", [filter_size, filter_size, x_shape[3], num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_conv", [num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        h = []
        result = []
        

        if not is_output:
            h = LeakyReLU(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b)
            #result = h
        else:
            h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b


        return h #result

def fc( x, out_size=50, is_output=False, name="fc" ):
    with tf.variable_scope(name) as scope:
        shape = x.get_shape().as_list()
        W = tf.get_variable("W_fc", [shape[1], out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("B_fc", [out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        
        h = []
        if not is_output:
            h = tf.nn.relu(tf.matmul(x,W)+b)
        else:
            h = tf.matmul(x,W)+b
        return h

def Generator(z):
    with tf.variable_scope("G_model") as scope:
        noise_2 = fc(z, out_size=4*4*1024,is_output=True)
        noise_3 = tf.reshape(noise_2,[BATCH_SIZE, 4, 4, 1024])
        G_1 = conv_t(noise_3,num_filters=512, stride=2,name="G1")
        print G_1.get_shape()
        G_2 = conv_t(G_1,num_filters=256, stride=2,name="G2")
        print G_2.get_shape()
        G_3 = conv_t(G_2,num_filters=128, stride=2,name="G3")
        print G_3.get_shape()
        G_4 = conv_t(G_3, num_filters=3, stride=2, is_output=True, name="G4")
        print G_4.get_shape()
        #G_5 = conv_t(G_4,num_filters=3, stride=2, is_output=True)
        #print G_5.get_shape()
	
        last = G_4
        last = tf.sigmoid(last)
        #shape = last.get_shape().as_list()
        #G_flat = tf.reshape(last, [-1,shape[1]*shape[2]*shape[3]])
        #print G_flat.get_shape()
        return last#G_flat


def Discriminator(input, reuse=False):
    with tf.variable_scope("D_model", reuse=reuse) as scope:
        #input = tf.reshape(input, [BATCH_SIZE, 32,32,3],'D_reshape')
        #print input.get_shape()
        D_1 = conv(input, num_filters=188, stride=2, name='D1')
        print D_1.get_shape()
        D_2 = conv(D_1,num_filters=256,stride=2,name='D2')
        print D_2.get_shape()
        D_3 = conv(D_2,num_filters=512,stride=2,name='D3')
        print D_3.get_shape()
        D_4 = conv(D_3,num_filters=1024,stride=2,name='D4')
        print D_4.get_shape()
     
        last = D_4
        #last = tf.sigmoid(D_4)
        shape = last.get_shape().as_list()
        D_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
        #D_flat = tf.tanh(D_flat)
        D_out = fc(D_flat, out_size=1, is_output=True, name='D_out')
        #D_out = LeakyReLU(D_out)
        #D_out = conv(last, num_filters =1, stride=shape[1], is_output=True,batch_norm=False)
        print D_out.get_shape()
        return D_out

z = tf.placeholder(tf.float32, [None, 100], name='z')
x_tilda = Generator(z)

shape_fd = x_tilda.get_shape().as_list()

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, shape_fd[1]* shape_fd[2]* shape_fd[3] ], name="input_data")
print 'real_data',real_data.get_shape()

x = tf.reshape(real_data, [BATCH_SIZE,shape_fd[1],shape_fd[2],shape_fd[3]])

print '------------------'

d_x_tilda = Discriminator(x_tilda)

d_x = Discriminator(x, reuse=True)

print '------------------'

epsilon = tf.random_uniform(
    shape=[BATCH_SIZE,1,1,1], 
    minval=0.,
    maxval=1.
)
x_hat = epsilon*x + (1- epsilon)*x_tilda

d_x_hat = Discriminator(x_hat, reuse=True) 


#gen_cost = -tf.reduce_mean(d_x_tilda)
#disc_cost = tf.reduce_mean(d_x_tilda) - tf.reduce_mean(d_x)

gradients = tf.gradients(d_x_hat, x_hat)[0]
print gradients.get_shape()

norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
print norms.get_shape()

grad_penalty = (norms-1.)**2

L = tf.reshape(d_x_tilda,[-1]) - tf.reshape(d_x, [-1]) + tf.reshape(grad_penalty, [-1])

disc_cost = tf.reduce_mean(L)
gen_cost = -tf.reduce_mean(d_x_tilda)

with tf.name_scope('Optimizer'):
    
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G_model")
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.999).minimize(gen_cost, var_list=g_vars)
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_model")
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.999).minimize(disc_cost, var_list=d_vars)


    
g_summary = tf.summary.scalar( 'G Cost', gen_cost )
d_summary = tf.summary.scalar( 'D Cost', disc_cost )

merged_summary_op = tf.summary.merge_all()

save_dir = "q"

summary_writer = tf.summary.FileWriter("./"+ save_dir,graph=sess.graph)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'p/1/1/lab7.ckpt')
'''
NUM_EPOCHS = 400000
n_critic = 5

for i in range(NUM_EPOCHS):
    batch = []
    z_in = []
    for j in range(n_critic):
        #z_in = np.ones([BATCH_SIZE, 100]) * 0.5
        z_in = np.random.rand(BATCH_SIZE, 100)
        batch = batch_utils.next_batch(BATCH_SIZE)

        disc_train_op.run(feed_dict={real_data:batch[0],z:z_in})
    gen_train_op.run(feed_dict={real_data:batch[0],z:z_in})
    if i % 10==0:
	print i
        summary_str,n_data,im_data,im_real,dxt = sess.run([merged_summary_op,z,x_tilda, x, d_x_tilda ],feed_dict={real_data:batch[0],z:z_in})
	#print im_data
	#print im_real
	#print dxt
	for iter, im in enumerate(im_data):
	    imsave(save_dir+'/'+str(iter)+'.png',im)
	#print im_data[0]
	#print im_data[0]*255.0
	#imsave(save_dir+'/Test_1.png',im_data[0])
	#imsave(save_dir+'/Test_2.png',im_data[0]*255.0)
	imsave(save_dir+'/real.png',im_real[0])
	summary_writer.add_summary(summary_str,i)
        
    if i% 100 ==0:
        saver.save(sess, save_dir+"/lab7.ckpt")
    


saver.save(sess, save_dir+"/lab7.ckpt")
summary_writer.close()
'''
### Iterpolating between 2 Z-values ###
steps = int(np.sqrt(BATCH_SIZE))
print steps
z_start = np.random.rand(steps, 100)
#print z_all.shape
z_out = np.random.rand(steps, 100)
z_all = []
for i in range(steps):
    z_row = []
    for j in range(100):
	if j == 0:
	    z_row = np.transpose(np.atleast_2d(np.linspace(z_start[i][j], z_out[i][j], steps)))
	    #print z_row.shape
	else:
	    z_temp = np.transpose(np.atleast_2d(np.linspace(z_start[i][j], z_out[i][j], steps)))
            z_row = np.concatenate((z_row,z_temp), 1)
    #print z_row.shape
    if i ==0 :
	z_all = z_row
    else:
    	z_all = np.concatenate((z_all,z_row),0)
    print z_all.shape
#print z_all.shape 

z_t,im_data = sess.run([z, x_tilda ],feed_dict={z:z_all})
for iter, im in enumerate(im_data):
    imsave(save_dir+'/'+str(iter)+'.png',im)


