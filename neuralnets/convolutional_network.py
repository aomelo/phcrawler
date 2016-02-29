'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
import input_data
import tensorflow as tf
import numpy as np
from input_data import DataSet, DataSets

# Network Parameters
#n_input = 784 # MNIST data input (img shape: 28*28)
#n_classes = 10 # MNIST total classes (0-9 digits)
n_classes = 8 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

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

image_height = image_height - (image_height % 4)
image_width = image_width - (image_width % 4)

train_images = crop(train_images, image_height, image_width)
test_images = crop(test_images, image_height, image_width)
valid_images = crop(valid_images, image_height, image_width)

mnist = DataSets()
mnist.train = DataSet(train_images, train_labels)
mnist.validation = DataSet(valid_images, valid_labels)
mnist.test = DataSet(test_images, test_labels)

n_input = image_height*image_width*image_depth  # MNIST data input (img shape: 28*28)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 1




# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, image_height, image_width, image_depth])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    print("reshape",str(_weights['wd1'].get_shape().as_list()[0]))
    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, image_depth, 32])), # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([image_height/4*image_width/4*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

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
