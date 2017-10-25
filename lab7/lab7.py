import numpy as np 
import tensorflow as tf
# import batch_utils

tf.reset_default_graph()
sess = tf.InteractiveSession()

batch_size = 5

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_t( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv_t" ):
    
    with tf.name_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        #W = weight_variable([filter_size, filter_size, x.get_shape().as_list()[3], num_filters])
        W = tf.Variable( 1e-3*np.random.randn(filter_size, filter_size, x_shape[3], x_shape[3]).astype(np.float32))
        b = bias_variable([x_shape[3]])
        h = []
        

        if not is_output:
            h = tf.nn.relu(tf.nn.conv2d_transpose(x, W, output_shape=[batch_size ,x_shape[1]*2 ,x_shape[2]*2 ,x_shape[3]], strides=[1,stride,stride,1], padding='SAME') + b)        
            h = conv(h,num_filters=num_filters,stride=1)
        else:
            h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME') + b
        return h

def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv" ):
    
    with tf.name_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        #W = weight_variable([filter_size, filter_size, x.get_shape().as_list()[3], num_filters])
        W = tf.Variable( 1e-3*np.random.randn(filter_size, filter_size, x_shape[3], num_filters).astype(np.float32))
        b = bias_variable([num_filters])
        h = []
        

        if not is_output:
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

def Generator(n_samples):
    with tf.name_scope('G_model') as g_scope:
        noise = tf.random_normal([batch_size, 10,10,1])
        #G_z = tf.placeholder(tf.float32, shape=[batch_size, 10, 10, 1], name="Zs")

        G_1 = conv(noise,num_filters=1024, stride=3)
        print G_1.get_shape()
        G_2 = conv_t(G_1,num_filters=512, stride=2)
        print G_2.get_shape()

        G_3 = conv_t(G_2,num_filters=256, stride=2)
        print G_3.get_shape()
        G_4 = conv_t(G_3,num_filters=188, stride=2)
        print G_4.get_shape()
        G_5 = conv_t(G_4,num_filters=3, stride=2, is_output=True)
        print G_5.get_shape()
        return G_5

def Discriminator(input):
    with tf.variable_scope('D_model', reuse=True):
        #d_scope.reuse_variables()
        D_1 = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 3])
        print D_1.get_shape()
        D_2 = conv(D_1,num_filters=188,stride=2)
        print D_2.get_shape()
        D_3 = conv(D_2,num_filters=256,stride=2)
        print D_3.get_shape()
        D_4 = conv(D_3,num_filters=512,stride=2)
        print D_4.get_shape()
        D_5 = conv(D_4,num_filters=1024,stride=2)
        print D_5.get_shape()
        shape = D_5.get_shape().as_list()
        D_flat = tf.reshape(D_5,[-1,shape[1]*shape[2]*shape[3]])
        D_out = fc(D_flat,out_size=100,is_output=True)
        print D_out.get_shape()
        return D_out

NUM_EPOCHS = 1#1000*2

n_critic = 5

fake_data = Generator(5)

print '------------------'

disc_fake = Discriminator(fake_data)

gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

alpha = tf.random_uniform(
    shape=[BATCH_SIZE/len(DEVICES),1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty


for j in range(NUM_EPOCHS):
    for t in range(n_critic):
        for i in range(batch_size):
            x = get_real_data()
            epsilon = np.random.random()
            x_bar = 1
            x_hat = epsilon * x + (1-epsilon)*x_bar

# G_3 = conv_t(G_2,num_filters=1024, stride=2)
# print G_3.get_shape()


#x_image = tf.reshape(x, [-1,512,512,3])

# with tf.name_scope('down1') as scope:
#     l1_h0 = conv(x_image,num_filters=64)
#     print l1_h0.get_shape()
#     l1_h1 = conv(l1_h0,num_filters=64)
#     print l1_h1.get_shape()
#     l1_h2 = conv(l1_h1,num_filters= 64)
#     print l1_h2.get_shape()

# with tf.name_scope('down2') as scope:
#     l2_h0 = max_pool_2x2(l1_h2)
#     print l2_h0.get_shape()
#     l2_h1 = conv(l2_h0,num_filters=128)
#     print l2_h1.get_shape()
#     l2_h2 = conv(l2_h1,num_filters=128)
#     print l2_h2.get_shape()

# with tf.name_scope('up1') as scope:
#     d1 = conv(l2_h2,filter_size=3, num_filters=l2_h2.get_shape().as_list()[3], transpose=True, stride=2)
#     print d1.get_shape()

#     out_shape = d1.get_shape().as_list()
#     in_shape = l1_h2.get_shape().as_list()
#     side = (in_shape[1]/2)-(out_shape[1]/2)


#     crop_l1_h2 = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img,side,side,out_shape[1],out_shape[1] ),l1_h2, name="crop1") 
#     print crop_l1_h2.get_shape()
    
#     d1_h0 = tf.concat([d1,crop_l1_h2],3)
#     print d1_h0.get_shape()

#     d1_h1 = conv(d1_h0,num_filters=64)
#     drop = tf.nn.dropout(d1_h1, keep_prob)
#     print drop.get_shape()
#     d1_h2 = conv(drop,is_output=True,num_filters=2)
#     print d1_h2.get_shape()



#     #shape = l2_h2.get_shape().as_list()
#     #flat = tf.reshape(l2_h2,[-1,shape[1]*shape[2]*shape[3]])
#     #fc2 = fc(flat,out_size=10,is_output=True)

# label_conv = tf.nn.softmax(d1_h2)

# with tf.name_scope('Cost') as scope:
#     cross_entropy = -tf.reduce_sum(label_ * tf.log(tf.clip_by_value(label_conv,1e-10,1)), axis=3)
#     cross_entropy =  tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cross_entropy, axis=2),axis=1)) 
 
# with tf.name_scope('Accuracy') as scope:
#     correct_prediction = tf.equal(tf.argmax(label_conv,3), tf.argmax(label_,3))
#     label_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
# with tf.name_scope('Optimizer') as scope:
#     train_step = tf.train.AdamOptimizer(3e-6).minimize(cross_entropy)
    
# acc_summary = tf.summary.scalar( 'Accuracy', label_acc )
# cost_summary = tf.summary.scalar( 'Cost', cross_entropy )

# merged_summary_op = tf.summary.merge_all()

# save_dir = "j"

# summary_writer = tf.summary.FileWriter("./"+ save_dir + "/train",graph=sess.graph)
# # validate_writer = tf.summary.FileWriter("./"+ save_dir +"/validate")

# saver = tf.train.Saver()

# sess.run(tf.global_variables_initializer())

# NUM_EPOCHS = 1#1000*2


# for i in range(NUM_EPOCHS):
    	
#     #batch = batch_utils.next_batch(batch_size)

#     # train_step.run(feed_dict={G_z:np.random.uniform(0.0,1.0,(5,10,10))})
#     summary_str = sess.run([merged_summary_op],feed_dict={G_z:np.random.uniform(0.0,1.0,(5,10,10,1))})
#     summary_writer.add_summary(summary_str,i)
     
#     # if i % 10 == 0:
    
#     #     print("Train: %d, %g"%(i,l))
	
#     # if i % 100 == 0:
#     #     test_batch = batch_utils.next_batch(batch_size,train=False)
#     #     summary_str,l = sess.run([merged_summary_op, label_acc],feed_dict={x_image :test_batch[0], label_:test_batch[1], keep_prob:.5})
        
#     #     print("Test: %d, %g"%(i,l))
#     # 	validate_writer.add_summary(summary_str,i)

# #saver.save(sess, save_dir+"/lab7.ckpt")

# summary_writer.close()
# # validate_write.close()


