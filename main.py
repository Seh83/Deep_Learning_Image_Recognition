import tensorflow as tf
import numpy as np

# Get DataSet
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# Setp: 1
training_digits, training_lables = mnist.train.next_batch(5000)
test_digits, test_lables = mnist.test.next_batch(200)

training_digits_pl = tf.placeholder("float", [None, 784])
test_digits_pl = tf.placeholder("float", [784])

# Sept: 2 Find the nearest distance between the test set and the input point.

l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digits_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get the minimum distance index
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initilizating the variables

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over the test data
    for i in range(len(test_digits)):
        # get the nearest neighbour
        nn_index = sess.run(pred, feed_dict={training_digits_pl: training_digits, test_digits_pl: test_digits[i, :]})

        # Get the nearest class label and compare it to true label
        print("Test:", i, "Prediction:", np.argmax(training_lables[nn_index]), "True_Lables:",
              np.argmax(test_lables[i]))

        # Calculate Accuracy
        if np.argmax(training_lables[nn_index] == np.argmax(test_lables[i])):
            accuracy += 1. / len(test_digits)
    print("Done!")
    print("Accuracy", accuracy)

