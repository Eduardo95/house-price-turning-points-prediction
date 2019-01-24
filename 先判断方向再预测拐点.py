# -*- coding: utf-8 -*-
import pandas as pd
import os
from imblearn.over_sampling import ADASYN
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
'''
这个文件使用一种新的方法来判断拐点，就是先判断下一时刻房价指数变动的方向，再结合上一时刻房价指数的变动方向，
获得这一时刻的状态（拐点或者非拐点）
'''
case_s_data = pd.read_pickle('data_directory\case_shiller_data_and_turning_points.pkl')
nytimes_data = pd.read_pickle('data_directory\\nytimes_data.pkl')
original_cs_index = case_s_data['case_shiller_original'][11:]  # 这是为了日期的对齐操作
nytimes_index_log_moving_avg_diff = nytimes_data['log_moving_avg_diff_index'][11:370]  # 这是为了日期的对齐操作
cs_index_log_moving_avg_diff = case_s_data['log_moving_avg_diff'][11:]  # 这是为了日期的对齐操作
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
这里计算原始凯斯西勒指数的变动方向，下一时刻比这一时刻值更高，则记为1，否则记为0
'''
direction = list()
for i in range(len(original_cs_index) - 1):
    if original_cs_index[i + 1] > original_cs_index[i]:
        direction.append(1)
    elif original_cs_index[i + 1] < original_cs_index[i]:
        direction.append(0)
direction.append(-1)
up_features = list()
down_features = list()
up_labels = list()
down_labels = list()
feature_window = 12

'''
网络结构
'''
training_epochs = 10000
batch_size = 10
display_step = 500
n_hidden_1 = 64
n_hidden_2 = 32
n_input = 12
n_classes = 2
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.001  # 初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=1000, decay_rate=0.9)
add_global = global_step.assign_add(1)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
train = tf.placeholder(tf.float32)
keep_prob = tf.placeholder("float")


def batch_norm_layer(value, train=None, name='batch_norm'):
    if train is not None:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=False)


def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)  # batch_norm_layer((layer_1), train))
    # layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)  # batch_norm_layer((layer_2), train))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(cost)
init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)

for i in range(328, 358, 1):
    train_features = list()
    train_labels = list()
    test_features = list()
    test_labels = list()
    test_features.append(cs_index_log_moving_avg_diff[i - 11:i + 1])
    test_labels.append(direction[i])
    for j in range(11, i, 1):
        train_features.append(cs_index_log_moving_avg_diff[j - 11:j + 1])
        train_labels.append(direction[j])
    train_features_ADASYN, train_labels_ADASYN = ADASYN().fit_sample(train_features, train_labels)
    test_labels_2 = list()
    train_labels_ADASYN_2 = list()
    for label in test_labels:
        test_labels_2.append([1, 0] if label == 0 else [0, 1])
    for label in train_labels_ADASYN:
        train_labels_ADASYN_2.append([1, 0] if label == 0 else [0, 1])
    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x: train_features_ADASYN, y: train_labels_ADASYN_2, keep_prob: 0.6})  # , train: 1})
        predict = sess.run(pred, feed_dict={x: test_features, keep_prob: 1})
        predict_t = -1
        if predict[0][0] > predict[0][1]:
            predict_t = 0
        else:
            predict_t = 1
        # print(predict_t)
        # print(test_labels)
        if direction[i - 1] == 0 and test_labels[0] == 0 and predict_t == 0:
            print("这是一个预测正确的不是拐点")
        elif direction[i - 1] == 0 and test_labels[0] == 1 and predict_t == 1:
            print("这是一个预测正确的波谷")
        elif direction[i - 1] == 1 and test_labels[0] == 1 and predict_t == 1:
            print("这是一个预测正确的不是拐点")
        elif direction[i - 1] == 1 and test_labels[0] == 0 and predict_t == 0:
            print("这是一个预测正确的波峰")
        else:
            print(test_labels[0], "预测：", predict_t, "这是一个预测错误的点")
# print(cs_index_log_moving_avg_diff)
# print(original_cs_index)
# print(direction)
# print(nytimes_index_log_moving_avg_diff)
