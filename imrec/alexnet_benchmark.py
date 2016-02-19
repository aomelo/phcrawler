# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Timing benchmark for AlexNet inference.
To run, use:
  bazel run -c opt --config=cuda \
      third_party/tensorflow/models/image/alexnet:alexnet_benchmark
Across 100 steps on batch size = 128.
Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch
Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cPickle as pickle
import numpy as np
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """Build the AlexNet model.
    Args:
      images: Images Tensor
    Returns:
      pool5: the last Tensor in the convolutional component of AlexNet.
      parameters: a list of Tensors corresponding to the weights and biases of the
          AlexNet model.
    """
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 1, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        print(images.get_shape().as_list())
        print(kernel.get_shape().as_list())
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    # lrn1
    # TODO(shlens, jiayq): Add a GPU version of local response normalization.
    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)
    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)
    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)
    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)
    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)
    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)
    return pool5, parameters


def time_tensorflow_run(session, target, info_string):
    """Run the computation to obtain the target tensor and print timing stats.
    Args:
      session: the TensorFlow session to run the computation under.
      target: the target Tensor that is passed to the session's run() function.
      info_string: a string summarizing this run, to be printed with the stats.
    Returns:
      None
    """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def run_benchmark():

    pickle_file = 'notMNIST.pickle'

    save = pickle.load(open(pickle_file, 'rb'))
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    """Run the benchmark on AlexNet."""
    with tf.Graph().as_default():

        # Generate some dummy images.
        image_height = 240
        image_width = 424
        image_depth = 1 # greyscale
        num_labels = 15
        batch_size = FLAGS.batch_size

        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, image_height, image_width, image_depth))
        tf_train_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        tf_valid_labels = tf.constant(valid_labels)
        tf_test_labels = tf.constant(test_labels)


        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.
        images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                               image_height,
                                               image_width,
                                               image_depth],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        # Build a Graph that computes the logits predictions from the
        # inference model.
        pool5, parameters = inference(tf_train_dataset)
        _, parameters_valid = inference(tf_valid_dataset)
        _, parameters_test = inference(tf_test_dataset)


        # Build an initialization operation.
        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        #config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        sess.run(init)

        loss = tf.reduce_mean(
            tf.nn.l2_loss(parameters, tf_train_labels))

         # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(parameters)
        valid_prediction = tf.nn.softmax(parameters_valid)
        test_prediction = tf.nn.softmax(parameters_test)

        num_steps = 3001
        print('Initialized')
        for step in range(num_steps):
            offset = (step * FLAGS.batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = sess.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

        # Run the forward benchmark.
        #time_tensorflow_run(sess, pool5, "Forward")
        # Add a simple objective so we can calculate the backward pass.
        #objective = tf.nn.l2_loss(pool5)
        # Compute the gradient with respect to all the parameters.
        #grad = tf.gradients(objective, parameters)
        # Run the backward benchmark.
        #time_tensorflow_run(sess, grad, "Forward-backward")






def main(_):
    run_benchmark()
if __name__ == '__main__':
    tf.app.run()