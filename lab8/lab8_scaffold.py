

import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq import sequence_loss, rnn_decoder

#
# -------------------------------------------
#
# Global variables

batch_size = 3 #50
sequence_length = 5# 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
#targ_ph = tf.placeholder( tf.float32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1)

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

# create a BasicLSTMCell
#   use it to create a MultiRNNCell
#   use it to create an initial_state
#     note that initial_state will be a *list* of tensors!


cell = BasicLSTMCell( state_dim )
cell2 = BasicLSTMCell( state_dim )
lstm = MultiRNNCell([cell, cell2])
initial_state = lstm.zero_state(batch_size, tf.float32)

# call seq2seq.rnn_decoder
with tf.variable_scope("decoder") as scope:
    outputs, final_state = rnn_decoder(inputs, initial_state, lstm)

# transform the list of state outputs to a list of logits.
# use a linear transformation.

W = tf.Variable(tf.random_normal([state_dim,vocab_size], stddev=0.02))
b = tf.Variable(tf.random_normal([vocab_size]))
logits = [tf.matmul(output, W) + b * batch_size for output in outputs]

l_shape = logits[0].get_shape().as_list()
loss_W = [1.0 for i in range(sequence_length)]

# call seq2seq.sequence_loss
loss = sequence_loss(logits, targets, loss_W)

# create a training op using the Adam optimizer
optim = tf.train.AdamOptimizer(1e-4).minimize(loss)

# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

# s_in_ph = tf.placeholder( tf.int32, [ 1, sequence_length ], name='inputs' )
# s_in_onehot = tf.one_hot( s_in_ph, vocab_size, name="input_onehot" )

# s_inputs = tf.split( s_in_onehot, sequence_length, axis=1 )
# s_inputs = [ tf.squeeze(input_, [1]) for input_ in s_inputs ]

s_inputs = [tf.placeholder(tf.int32, [1])]
# s_inputs = [ tf.squeeze(input_, [1]) for input_ in s_inputs ]
s_initial_state = lstm.zero_state(1, tf.float32)

# call seq2seq.rnn_decoder
with tf.variable_scope("s_decoder") as scope:
    s_outputs, s_final_state = rnn_decoder(s_inputs, s_initial_state, lstm)



#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    print prime[:-1]
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        #feed = { s_inputs:x }
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        #feed = { s_inputs:x }
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(10):
    print 'here'
    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()
        print x,y

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

#    print sample( num=60, prime="And " )
    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
