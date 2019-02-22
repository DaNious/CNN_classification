import tensorflow as tf
import numpy as np
from data_input import import_data
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

"""
Here defines the hyper-parameters
"""
learning_rate = 0.0001
batch_size = 50

input_dim = 1024 # 8*128
num_classes = 5

#split_ratio = 0.8 # percentage or training set
num_train = 100

num_steps = 2000

display_step = 10

dropout = 0.75  # Dropout, probability to keep units

Image = tf.placeholder(tf.float32, [None, input_dim])
Classes = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


def maxpool2d(x, k):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')


def feature_extractor(x, weights, biases, dropout):
    # data input is a 1-D vector of 1024 features (8*128 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 32, 32, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, 2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, 2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out1'])

    return conv1, conv2, out


# Store layers weight & bias
weights = {
    #CNN
    'wc1': tf.get_variable('wc1',shape=[5,5,1,32],initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('wc2',shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('wd1',shape=[5*5*64,1024],initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('out',shape=[1024,num_classes],initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    #CNN
    'bc1': tf.get_variable('bc1',shape=[32],initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('bc2',shape=[64],initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('bd1',shape=[1024],initializer=tf.contrib.layers.xavier_initializer()),
    'out1': tf.get_variable('out1',shape=[num_classes],initializer=tf.contrib.layers.xavier_initializer())
}

conv1,conv2, logits = feature_extractor(Image, weights, biases, keep_prob)
prediction = tf.nn.sigmoid(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Classes))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Start
# The training data is 1-1
# The test data is 16-1
source_train_x, source_train_y, source_test_x, source_test_y = import_data(num_train)


def next_batch_idx(batch_size, epoch, index):
    if epoch == 0:
        np.random.shuffle(index)
    start = (epoch*batch_size)%(index.shape[0])
    end = start + batch_size
    if(end >= (index.shape[0])):
        np.random.shuffle(index)
        start = 0
        end = start + batch_size
    idx = index[start:end]
    return idx, index


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    train_index = np.arange(num_classes * num_train)
    test_index = np.arange(113)
    sess.run(init)

    for step in range(1, num_steps+1):
        train_idx, train_index = next_batch_idx(batch_size, step, train_index)
        source_batch = source_train_x[train_idx]
        source_label_batch = source_train_y[train_idx]

        test_idx, test_index = next_batch_idx(batch_size, step, test_index)
        test_batch = source_test_x[test_idx]
        test_label_batch = source_test_y[test_idx]

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={Image: source_batch, Classes: source_label_batch, keep_prob: 0.8})

        # image1 = source_train_x[1].reshape(1,input_dim)
        # conv1_ = sess.run(conv1, feed_dict={Image: image1, Classes: source_label_batch})
        # conv2_ = sess.run(conv2, feed_dict={Image: image1, Classes: source_label_batch})
        # conv1_ = np.sum(conv1_, axis=0)
        # conv2_ = np.sum(conv2_, axis=0)
        # print(conv1_.shape)
        # print(conv2_.shape)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={Image: source_batch,
                                                                Classes: source_label_batch,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc*100), '%')
            print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={Image: test_batch,
                                                Classes: test_label_batch,
                                                keep_prob: 1.0})*100, '%')

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={Image: source_test_x,
                                      Classes: source_test_y,
                                    keep_prob: 1.0})*100, '%')

    # for i in range(20):
    #     image1 = source_train_x[i].reshape(1, input_dim)
    #     conv1_ = sess.run(conv1, feed_dict={Image: image1, Classes: source_label_batch})
    #     conv2_ = sess.run(conv2, feed_dict={Image: image1, Classes: source_label_batch})
    #     conv1_ = np.sum(conv1_, axis=0)
    #     conv2_ = np.sum(conv2_, axis=0)
    #     name = 'conv1_' + str(0) + str(source_label_batch[i].tolist().index(1))
    #     tmp1_ = conv1_[:, :, 0]
    #     plt.imshow(tmp1_, cmap='Greys_r')
    #     plt.savefig(name)
    #     name = 'conv2_' + str(0) + str(source_label_batch[i].tolist().index(1))
    #     tmp2_ = conv2_[:, :, 0]
    #     plt.imshow(tmp2_, cmap='Greys_r')
    #     plt.savefig(name)


    # for i in range(32):
    #     name = 'conv1_' + str(i)
    #     tmp1_ = conv1_[:, :, i]
    #     plt.imshow(tmp1_, cmap='Greys_r')
    #     plt.savefig(name)
    # for i in range(64):
    #     name = 'conv2_' + str(i)
    #     tmp2_ = conv2_[:, :, i]
    #     plt.imshow(tmp2_, cmap='Greys_r')
    #     plt.savefig(name)

    print("Output Finished!")
