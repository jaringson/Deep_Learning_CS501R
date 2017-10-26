import numpy as np 
import tensorflow as tf
import batch_utils

tf.reset_default_graph()
sess = tf.InteractiveSession()

BATCH_SIZE = 5
LAMBDA = 10 # Gradient penalty lambda hyperparameter

g_scope = 0
d_scope = 0

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

def conv_t( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv_t" ):
    
    with tf.name_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        #W = weight_variable([filter_size, filter_size, x.get_shape().as_list()[3], num_filters])
        W = tf.Variable( 1e-3*np.random.randn(filter_size, filter_size, x_shape[3], x_shape[3]).astype(np.float32))
        b = bias_variable([x_shape[3]])
        h = []
        

        if not is_output:
            h = tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=[BATCH_SIZE ,x_shape[1]*2 ,
                    x_shape[2]*2 ,x_shape[3]], strides=[1,stride,stride,1], padding='SAME') + b)        
            
        else:
            h = tf.nn.conv2d_transpose(x, W, output_shape=[BATCH_SIZE ,x_shape[1]*2 ,
                    x_shape[2]*2 ,x_shape[3]], strides=[1,stride,stride,1], padding='SAME') + b
        h = conv(h,num_filters=num_filters,stride=1)

        mean, var = tf.nn.moments(h, [0,1,2], keep_dims=False)
        offset = np.zeros(mean.get_shape()[-1], dtype='float32')
        scale = np.ones(var.get_shape()[-1], dtype='float32')
        result = tf.nn.batch_normalization(h, mean, var, offset, scale, 1e-4)

        return result

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv" ):
    with tf.name_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        #W = weight_variable([filter_size, filter_size, x.get_shape().as_list()[3], num_filters])
        W = tf.Variable( 1e-3*np.random.randn(filter_size, filter_size, x_shape[3], 
            num_filters).astype(np.float32))
        b = bias_variable([num_filters])
        h = []
        

        if not is_output:
            h = LeakyReLU(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b)
        else:
            h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b

        mean, var = tf.nn.moments(h, [0,1,2], keep_dims=False)
        offset = np.zeros(mean.get_shape()[-1], dtype='float32')
        scale = np.ones(var.get_shape()[-1], dtype='float32')
        result = tf.nn.batch_normalization(h, mean, var, offset, scale, 1e-4)
        
        return result

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

def Generator(n_samples):
    global g_scope
    with tf.variable_scope('G_model') as g_scope:
        noise = tf.random_normal([BATCH_SIZE, 10,10,1])
        #G_z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10, 10, 1], name="Zs")

        G_1 = conv(noise,num_filters=1024, stride=3)
        print G_1.get_shape()
        G_2 = conv_t(G_1,num_filters=512, stride=2)
        print G_2.get_shape()
        G_3 = conv_t(G_2,num_filters=3, stride=2)
        print G_3.get_shape()
        #G_4 = conv_t(G_3,num_filters=188, stride=2)
        #print G_4.get_shape()
        #G_5 = conv_t(G_4,num_filters=3, stride=2, is_output=True)
        #print G_5.get_shape()
	
	last = G_3
        shape = last.get_shape().as_list()
        G_flat = tf.reshape(last, [-1,shape[1]*shape[2]*shape[3]])
        print G_flat.get_shape()
        return G_flat

def Discriminator(input):
    global d_scope
    with tf.variable_scope('D_model') as d_scope:
        #D_1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 3])
        input = tf.reshape(input, [BATCH_SIZE, 16,16,3],'D_reshape')
        print input.get_shape()
        D_2 = conv(input,num_filters=188,stride=2,name='D2')
        print D_2.get_shape()
        D_3 = conv(D_2,num_filters=256,stride=2,name='D3')
        print D_3.get_shape()
        #D_4 = conv(D_3,num_filters=512,stride=2)
        #print D_4.get_shape()
        #D_5 = conv(D_4,num_filters=1024,stride=2)
        #print D_5.get_shape()
        
	last = D_3
	shape = last.get_shape().as_list()
        D_flat = tf.reshape(last,[-1,shape[1]*shape[2]*shape[3]])
        D_out = fc(D_flat, out_size=1, is_output=True,name='D_out')
        d_scope.reuse_variables()
        print D_out.get_shape()
        return D_out


fake_data = Generator(5)
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, fake_data.get_shape()[1]], name="input_data")

print '------------------'

disc_fake = Discriminator(fake_data)

with tf.variable_scope(d_scope, reuse=True)

disc_real = Discriminator(real_data)

gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)


print '------------------'

alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
print differences.get_shape()
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty


with tf.name_scope('Optimizer'):
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=g_scope.name)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                          var_list=g_vars)
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=d_scope.name)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                           var_list=d_vars)


    
g_summary = tf.summary.scalar( 'G Cost', gen_cost )
d_summary = tf.summary.scalar( 'D Cost', disc_cost )

merged_summary_op = tf.summary.merge_all()

save_dir = "a"

summary_writer = tf.summary.FileWriter("./"+ save_dir,graph=sess.graph)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

NUM_EPOCHS = 1000*2
n_critic = 5

for i in range(NUM_EPOCHS):
    batch = []
    for i in range(n_critic):
        batch = batch_utils.next_batch(BATCH_SIZE)

        disc_train_op.run(feed_dict={real_data:batch[0]})
    gen_train_op.run(feed_dict={real_data:batch[0]})
    if i % 10:
	print i
        summary_str = sess.run([merged_summary_op],feed_dict={real_data:batch[0]})
        summary_writer.add_summary(summary_str[0],i)
     

saver.save(sess, save_dir+"/lab7.ckpt")

summary_writer.close()



