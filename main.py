import tensorflow as tf
import numpy as np
import input_data

mnist_width = 28
label_len = 10
n_visible = mnist_width * mnist_width + label_len  # mix input and labels
n_hidden = 500
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X

    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
    return [Y,Z]

# build model graph
Y,Z = model(X, mask, W, b, W_prime, b_prime)

cost = tf.reduce_sum(tf.pow(X - Z,2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# concatenate input with label
def concat(tr_x, tr_y, start, end):
	N, L = tr_x.shape
	L += label_len
	batch = np.zeros([N,L])
	for i in range(start,end+1):
		batch[i] = np.concatenate((tr_x[i],tr_y[i]))
	return batch

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tr_xy = concat(trX,trY,0,len(trX)-1)
    te_xy = concat(teX,teY,0,len(teX)-1)
    #print(tr_xy.shape,trX.shape,trY.shape)
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = tr_xy[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})
            #if(end % 10 == 0):
            #	print(sess.run(cost, feed_dict={X: input_, mask: mask_np}))
        mask_np = np.random.binomial(1, 1 - corruption_level, te_xy.shape)
        print(i, sess.run(cost, feed_dict={X: te_xy, mask: mask_np}))

