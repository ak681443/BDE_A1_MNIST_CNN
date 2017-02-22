# In[0]:
import tensorflow as tf
import numpy as np
import csv

sess = tf.InteractiveSession()

features = []

def append_new_weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def append_new_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def append_conv_layer(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def append_max_pool_layer(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, shape=[None, 784])
Y_ = tf.placeholder(tf.float32, shape=[None, 10])

#legacy variables
is_train = tf.placeholder(tf.bool)
W_OLD = tf.Variable(tf.zeros([784,10]))
B_OLD = tf.Variable(tf.zeros([10]))


W_C1 = append_new_weight([5, 5, 1, 32])
B_C1 = append_new_bias([32])

X_Bitmap = tf.reshape(X, [-1,28,28,1])


A_C1 = tf.nn.relu(append_conv_layer(X_Bitmap, W_C1) + B_C1)
A_P1 = append_max_pool_layer(A_C1)


W_C2 = append_new_weight([5, 5, 32, 64])
B_C2 = append_new_bias([64])

A_C2 = tf.nn.relu(append_conv_layer(A_P1, W_C2) + B_C2)
A_P2 = append_max_pool_layer(A_C2)


W_FC1 = append_new_weight([7 * 7 * 64, 1024])
B_FC1 = append_new_bias([1024])

A_P3_FLAT = tf.reshape(A_P2, [-1, 7*7*64])
A_FC1 = tf.nn.relu(tf.matmul(A_P3_FLAT, W_FC1) + B_FC1)

dropout_prob = tf.placeholder(tf.float32)
A_FC1_drop = tf.nn.dropout(A_FC1, dropout_prob)

W_FC2 = append_new_weight([1024, 10])
B_FC2 = append_new_bias([10])

Y_C = tf.matmul(A_FC1_drop, W_FC2) + B_FC2



saver = tf.train.Saver()
save_path="./back/model2.ckpt.8000"
saver.restore(sess, save_path)
print("Model restored.")

idx = 1
op_file = open('op_r', 'w')
with open('/home/arvind/Coursework/BDE/LABS/LAB1/test.csv', 'rb') as csvfile:
    mnist1 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in mnist1:
      py = sess.run(Y_C, feed_dict={X:[np.array(row).astype('float32')], dropout_prob:1.0})
      op_file.write("{},{}\n".format(idx,np.argmax(py[0],0)))
      idx =  idx + 1
