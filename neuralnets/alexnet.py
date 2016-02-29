'''
AlexNet implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
AlexNet Paper (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
import input_data
import numpy as np
import tensorflow as tf
from input_data import DataSet
from input_data import DataSets

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 1
#image_height = 28
#image_width = 28


# Network Parameters
#n_classes = 10 # MNIST total classes (0-9 digits)
n_classes = 8 # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units


def format1d(images):
    return images.reshape(images.shape[0],images.shape[1]*images.shape[2])

def crop(images,image_height,image_width):
    diff_height = images.shape[1] - image_height
    diff_width =  images.shape[2] - image_width
    offset_x = int(diff_height/2)
    offset_y = int(diff_width/2)
    return images[:,offset_x:offset_x+image_height,offset_y:offset_y+image_width]


#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#mnist = input_data.read_data_sets_ph("../../../imrec/notMNIST.pickle")
mnist = input_data.read_data_sets_ph("ph.h5")

image_height = 240
image_width = 424
image_depth = 3

print(mnist.train.images.shape)
train_images = np.reshape(mnist.train.images, (-1,image_height,image_width,image_depth))
test_images = np.reshape(mnist.test.images, (-1,image_height,image_width,image_depth))
valid_images = np.reshape(mnist.validation.images, (-1,image_height,image_width,image_depth))
train_labels = mnist.train.labels
test_labels = mnist.test.labels
valid_labels = mnist.validation.labels

image_height = image_height - (image_height % 8)
image_width = image_width - (image_width % 8)

train_images = crop(train_images, image_height, image_width)
test_images = crop(test_images, image_height, image_width)
valid_images = crop(valid_images, image_height, image_width)

mnist = DataSets()
mnist.train = DataSet(train_images, train_labels)
mnist.validation = DataSet(valid_images, valid_labels)
mnist.test = DataSet(test_images, test_labels)

n_input = image_height*image_width*image_depth  # MNIST data input (img shape: 28*28)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, image_height, image_width, image_depth])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']

    print('out' + str(out.get_shape().as_list()))
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, image_depth, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([image_height/8*image_width/8*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = alex_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.validation.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.labels.shape)
print(mnist.validation.labels.shape)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    test_size = mnist.test.images.shape[0]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size], keep_prob: 1.})
