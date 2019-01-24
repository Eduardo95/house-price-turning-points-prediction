# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.cross_validation import train_test_split
import math
from tensorflow.contrib.layers.python.layers import batch_norm
import os
'''
这个文件利用纽约时报的情绪数据和凯斯西勒指数的前滞窗口数据进行预测
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
case_s_data = pd.read_pickle('data_directory\case_shiller_data_and_turning_points.pkl')
nytimes_data = pd.read_pickle('data_directory\\nytimes_data.pkl')


def get_data(end):
    log_moving_avg_diff = case_s_data['log_moving_avg_diff'][11:end]
    tp_1_month = case_s_data['tp_class_one_month'][11:end]
    emo_index = nytimes_data['log_moving_avg_diff_index'][11:end]
    article_nums = nytimes_data['log_moving_avg_diff_nums'][11:end]
    # classification problem
    features = list()  # features
    feature_window = 12
    for i in range(len(log_moving_avg_diff) - feature_window + 1):
        feature = list()
        for j in range(6):  # feature_window
            feature.append(log_moving_avg_diff[i + j + 6])
        feature.append(emo_index[i + feature_window - 1 - 6])
        feature.append(emo_index[i + feature_window - 1 - 5])
        feature.append(emo_index[i + feature_window - 1 - 4])
        feature.append(emo_index[i + feature_window - 1 - 3])
        feature.append(emo_index[i + feature_window - 1 - 2])
        feature.append(emo_index[i + feature_window - 1 - 1])
        feature.append(article_nums[i + feature_window - 1 - 6])
        feature.append(article_nums[i + feature_window - 1 - 5])
        feature.append(article_nums[i + feature_window - 1 - 4])
        feature.append(article_nums[i + feature_window - 1 - 3])
        feature.append(article_nums[i + feature_window - 1 - 2])
        feature.append(article_nums[i + feature_window - 1 - 1])
        features.append(feature)
    label = tp_1_month[feature_window - 1:]
    test_feature = list()
    feature = list()
    for i in range(6):  # feature_window
        feature.append(case_s_data['log_moving_avg_diff'][end + 1 + i - feature_window + 6])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 6])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 5])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 4])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 3])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 2])
    feature.append(emo_index[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 1])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 6])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 5])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 4])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 3])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 2])
    feature.append(article_nums[len(log_moving_avg_diff) - feature_window + 1 + feature_window - 1 - 1])
    test_feature.append(feature)
    test_label = list()
    if case_s_data['tp_class_one_month'][end] == 0:
        test_label.append([1, 0, 0])
    elif case_s_data['tp_class_one_month'][end] == 1:
        test_label.append([0, 1, 0])
    else:
        test_label.append([0, 0, 1])
    x_re_sampled_smote, y_re_sampled_smote = SMOTE().fit_sample(features, label)
    # x_re_sampled_adasyn, y_re_sampled_adasyn = ADASYN().fit_sample(features, label)
    labels = list()
    for j in range(len(y_re_sampled_smote)):
        if y_re_sampled_smote[j] == 0:
            labels.append([1, 0, 0])
        elif y_re_sampled_smote[j] == 1:
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])

    train_feature, val_feature, train_label, val_label = train_test_split(x_re_sampled_smote, labels, random_state=0)
    return train_feature, val_feature, train_label, val_label, x_re_sampled_smote, labels, test_feature, test_label


# learning_rate = 0.001
training_epochs = 10000
batch_size = 10
display_step = 500
n_hidden_1 = 64
n_hidden_2 = 32
n_input = 18
n_classes = 3
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.001  # 初始学习率
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps=1000, decay_rate=0.9)
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
# 初始化变量
init = tf.global_variables_initializer()
result = list()
total = 0
config = tf.ConfigProto(allow_soft_placement=True)
for num in range(50):
    train_data_x, val_data_x, train_data_y, val_data_y, all_train_data_x, all_train_data_y, test_data_x, test_data_y = \
        get_data(num + 318)
    # 启动session
    print(test_data_y)
    times = 0
    with tf.Session(config=config) as sess:
        sess.run(init)
        # 启动循环开始训练
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = math.ceil(len(all_train_data_x) / batch_size)
            # 遍历全部数据集
            for k in range(total_batch - 1):
                batch_x = all_train_data_x[k * batch_size: (k + 1) * batch_size]
                batch_y = all_train_data_y[k * batch_size: (k + 1) * batch_size]
                # Run optimization op (back propagation) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})# , train: 1})
                # Compute average loss
                avg_cost += c / total_batch
            batch_x = all_train_data_x[total_batch - 1:]
            batch_y = all_train_data_y[total_batch - 1:]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})# , train: 1})
            avg_cost += c / total_batch
            # 显示训练中的详细信息
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            if avg_cost < 0.1:
                times += 1
            else:
                times = 0
            if times >= 10:
                break
        print(" Finished!")

        predict = sess.run(pred, feed_dict={x: test_data_x, keep_prob: 1})
        result.append(predict)
        print(predict)
        c_new = sess.run(cost, feed_dict={x:test_data_x, y:test_data_y, keep_prob: 1})
        print(c_new)
        # 测试 model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        t_or_f = accuracy.eval({x: test_data_x, y: test_data_y, keep_prob: 1})
        print("Accuracy:", t_or_f)
        if int(t_or_f) == 1:
            total += 1

print(total)
print(result)
