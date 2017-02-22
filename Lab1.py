# In[0]:
import tensorflow as tf
import numpy as np
import gc


sess = tf.InteractiveSession()

import csv
features = []
labels = []
#Coursework/BDE/LABS/LAB1
#read training set
with open('/home/arvind/Downloads/train.csv', 'rb') as csvfile:
    mnistdata = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in mnistdata:
        ohe = np.zeros(10)
        ohe[int(row[0])] = 1
        labels.append(ohe)
        features.append(np.array(row[1:]).astype(int))

#keep a validation step
features = np.array(features[:-200])
labels = np.array(labels[:-200])
features_test = np.array(features[-200:])
labels_test = np.array(labels[-200:])

def augument_data(images, labels):
	afine_tf = transform.AffineTransform(rotation=-0.3, shear = 0.1, translation=0.3)
	modified = transform.warp(v.astype(float), affne_tf)
#shuffle iterator
def shuffle_iterator():
    batch_id = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        size = 64
        for batch_id in range(0, len(features), size):
            images_batch = shuf_features[batch_id:batch_id+size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_id:batch_id+size]
            yield (images_batch, labels_batch)

def append_new_weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def append_new_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def append_conv_layer(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def append_max_pool_layer(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, shape=[None, 784])
Y_ = tf.placeholder(tf.float32, shape=[None, 10])

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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_C))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_C,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

save_path="./model3.ckpt.9000"
#saver.restore(sess, save_path)
print("Model restored.")


iterator = shuffle_iterator()
for i in range(10000):
  batch_x, batch_y = iterator.next()
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={X:features_test, Y_: labels_test, dropout_prob: 1.0})
    print("step %d, training accuracy %f"%(i, train_accuracy))
    #save_path = saver.save(sess, "./model4.ckpt.9000".format(i))
  train_step.run(feed_dict={X: batch_x, Y_: batch_y, dropout_prob: 0.5})

print 'training done'
del labels
del features
gc.collect()

idx = 1
op_file = open('op', 'w')
features = []
with open('/home/arvind/Coursework/BDE/LABS/LAB1/test.csv', 'rb') as csvfile:
    mnist1 = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in mnist1:
      py = sess.run(Y_C, feed_dict={X:[np.array(row).astype('float32')], dropout_prob:1.0})
      op_file.write("{},{}\n".format(idx,np.argmax(py[0],0)))
      idx =  idx + 1
