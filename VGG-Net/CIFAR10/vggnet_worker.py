"""
2021/9/16
author: tmq
code: cifar10_worker-v2
updated:
         1. add the condition of convergence.
         2. add the move average model for test
"""

import tensorflow as tf
from socket import *
import numpy as np
import pickle as pk
import math
from input_data import Cifar10
import time
import seblock
from init_parameters import create_placeholder
from mixup import mixup_data


# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 10

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 128
CONV3_SIZE = 3

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 128
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 256
CONV5_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV6_DEEP = 256
CONV6_SIZE = 3

# 第7层卷积层的尺寸和深度
CONV7_DEEP = 256
CONV7_SIZE = 3

# 第8层卷积层的尺寸和深度
CONV8_DEEP = 256
CONV8_SIZE = 3

# SEnet权重参数
# SEnet插入第一层卷积层的参数
SE1_SIZE = 1
SE1_CHANEL = 64
SE1_DEEP = 16

SE2_SIZE = 1
SE2_CHANEL = 16
SE2_DEEP = 64

# SEnet插入第二层卷积层的参数
SE3_SIZE = 1
SE3_CHANEL = 64
SE3_DEEP = 16

SE4_SIZE = 1
SE4_CHANEL = 16
SE4_DEEP = 64

# SEnet插入第三层卷积层的参数
SE5_SIZE = 1
SE5_CHANEL = 128
SE5_DEEP = 16

SE6_SIZE = 1
SE6_CHANEL = 16
SE6_DEEP = 128

# SEnet插入第四层卷积层的参数
SE7_SIZE = 1
SE7_CHANEL = 128
SE7_DEEP = 16

SE8_SIZE = 1
SE8_CHANEL = 16
SE8_DEEP = 128

# SEnet插入第五层卷积层的参数
SE9_SIZE = 1
SE9_CHANEL = 256
SE9_DEEP = 16

SE10_SIZE = 1
SE10_CHANEL = 16
SE10_DEEP = 256

# SEnet插入第六层卷积层的参数
SE11_SIZE = 1
SE11_CHANEL = 256
SE11_DEEP = 16

SE12_SIZE = 1
SE12_CHANEL = 16
SE12_DEEP = 256

# SEnet插入第七层卷积层的参数
SE13_SIZE = 1
SE13_CHANEL = 256
SE13_DEEP = 8

SE14_SIZE = 1
SE14_CHANEL = 8
SE14_DEEP = 256

# SEnet插入第八层卷积层的参数
SE15_SIZE = 1
SE15_CHANEL = 256
SE15_DEEP = 8

SE16_SIZE = 1
SE16_CHANEL = 8
SE16_DEEP = 256

# 正则化系数
REGULARIZERATION = 0.01

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.9, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 19020433, """Number of parameters byte.""")


# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '10.1.2.95', '''The ip address of parameter server''')


def forward_propagation(input_tensor, regularizer):

    is_train = tf.placeholder_with_default(False, (), 'is_train')
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv1 = tf.layers.batch_normalization(tf.nn.bias_add(conv1, conv1_biases), training=is_train)
        relu1 = tf.nn.relu(bn_conv1)
        se_conv1 = seblock.SE_block(relu1, ratio=4)

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(se_conv1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv2 = tf.layers.batch_normalization(tf.nn.bias_add(conv2, conv2_biases), training=is_train)
        relu2 = tf.nn.relu(bn_conv2)
        se_conv2 = seblock.SE_block(relu2, ratio=4)

    with tf.variable_scope('layer3-pool1'):
        pool1 = tf.nn.avg_pool(se_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer4-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv3 = tf.layers.batch_normalization(tf.nn.bias_add(conv3, conv3_biases), training=is_train)
        relu3 = tf.nn.relu(bn_conv3)
        se_conv3 = seblock.SE_block(relu3, ratio=8)

    with tf.variable_scope('layer5-conv4'):
        conv4_weights = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

        conv4 = tf.nn.conv2d(se_conv3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv4 = tf.layers.batch_normalization(tf.nn.bias_add(conv4, conv4_biases), training=is_train)
        relu4 = tf.nn.relu(bn_conv4)
        se_conv4 = seblock.SE_block(relu4, ratio=8)

    with tf.variable_scope('layer6-pool2'):
        pool2 = tf.nn.avg_pool(se_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer7-conv5'):
        conv5_weights = tf.get_variable("weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))

        conv5 = tf.nn.conv2d(pool2, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv5 = tf.layers.batch_normalization(tf.nn.bias_add(conv5, conv5_biases), training=is_train)
        relu5 = tf.nn.relu(bn_conv5)
        se_conv5 = seblock.SE_block(relu5, ratio=16)

    with tf.variable_scope('layer8-conv6'):
        conv6_weights = tf.get_variable("weight", [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [CONV6_DEEP], initializer=tf.constant_initializer(0.0))

        conv6 = tf.nn.conv2d(se_conv5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv6 = tf.layers.batch_normalization(tf.nn.bias_add(conv6, conv6_biases), training=is_train)
        relu6 = tf.nn.relu(bn_conv6)
        se_conv6 = seblock.SE_block(relu6, ratio=16)

    with tf.variable_scope('layer9-pool3'):
        pool3 = tf.nn.avg_pool(se_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer10-conv7'):
        conv7_weights = tf.get_variable("weight", [CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable("bias", [CONV7_DEEP], initializer=tf.constant_initializer(0.0))

        conv7 = tf.nn.conv2d(pool3, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv7 = tf.layers.batch_normalization(tf.nn.bias_add(conv7, conv7_biases), training=is_train)
        relu7 = tf.nn.relu(bn_conv7)
        se_conv7 = seblock.SE_block(relu7, ratio=32)

    with tf.variable_scope('layer11-conv8'):
        conv8_weights = tf.get_variable("weight", [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv8_biases = tf.get_variable("bias", [CONV8_DEEP], initializer=tf.constant_initializer(0.0))

        conv8 = tf.nn.conv2d(se_conv7, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv8 = tf.layers.batch_normalization(tf.nn.bias_add(conv8, conv8_biases), training=is_train)
        relu8 = tf.nn.relu(bn_conv8)
        se_conv8 = seblock.SE_block(relu8, ratio=32)

    with tf.variable_scope('layer12-pool4'):
        pool4 = tf.nn.avg_pool(se_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    P = tf.contrib.layers.flatten(pool4)

    with tf.variable_scope('layer13-fc'):
        fc_weights = tf.get_variable('weight', [P.shape[1], NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization = regularizer(fc_weights)
        fc_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(P, fc_weights) + fc_biases

    return logits, regularization, is_train


def compute_cost(logits, label_a, label_b, lam, regularization, learning_rate):

    cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label_a, 1))
    cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label_b, 1))
    cross_entropy_mean_a = tf.reduce_mean(cross_entropy_a)
    cross_entropy_mean_b = tf.reduce_mean(cross_entropy_b)
    loss = cross_entropy_mean_a*lam + cross_entropy_mean_b*(1 - lam) + regularization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)

    return logits, loss, grads_and_vars


def compute_accuracy(logits, labels):
    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    return accuracy


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def worker_random_minibatches(minibatch, worker_minibatch_size, seed):
    worker_batches = []
    for batch in minibatch:
        batches = random_mini_batches(batch[0], batch[1], worker_minibatch_size, seed)
        worker_batches.append(batches)

    return worker_batches


def load_dataset():

    cifar10 = Cifar10(path=r"./cifar-10-batches-py", one_hot=True)
    cifar10._load_data()

    # 准备训练集
    train_xs = cifar10.images / 255.0
    train_labels = cifar10.labels

    # 准备测试集
    np.random.seed(1)
    permutation = list(np.random.permutation(10000))
    shuffled_tx = cifar10.test.images[permutation, :, :, :] / 255.0
    shuffled_ty = cifar10.test.labels[permutation, :]

    test_feeds = []
    for k in range(10):
        test_xs = shuffled_tx[k * 1000:k * 1000 + 1000, :, :, :]
        test_labels = shuffled_ty[k * 1000:k * 1000 + 1000, :]

        test_feed = (test_xs, test_labels)

        test_feeds.append(test_feed)

    return train_xs, train_labels, test_feeds


def convert_dict_to_tuple(parameters_dict):

    dic = parameters_dict
    tuple = (
             dic['w1'], dic['b1'], dic['gama1'], dic['beta1'], dic['se1_w1'], dic['se1_b1'], dic['se1_w2'], dic['se1_b2'],
             dic['w2'], dic['b2'], dic['gama2'], dic['beta2'], dic['se2_w1'], dic['se2_b1'], dic['se2_w2'], dic['se2_b2'],
             dic['w3'], dic['b3'], dic['gama3'], dic['beta3'], dic['se3_w1'], dic['se3_b1'], dic['se3_w2'], dic['se3_b2'],
             dic['w4'], dic['b4'], dic['gama4'], dic['beta4'], dic['se4_w1'], dic['se4_b1'], dic['se4_w2'], dic['se4_b2'],
             dic['w5'], dic['b5'], dic['gama5'], dic['beta5'], dic['se5_w1'], dic['se5_b1'], dic['se5_w2'], dic['se5_b2'],
             dic['w6'], dic['b6'], dic['gama6'], dic['beta6'], dic['se6_w1'], dic['se6_b1'], dic['se6_w2'], dic['se6_b2'],
             dic['w7'], dic['b7'], dic['gama7'], dic['beta7'], dic['se7_w1'], dic['se7_b1'], dic['se7_w2'], dic['se7_b2'],
             dic['w8'], dic['b8'], dic['gama8'], dic['beta8'], dic['se8_w1'], dic['se8_b1'], dic['se8_w2'], dic['se8_b2'],
             dic['w9'], dic['b9'])
    return tuple


def replace_trainable_vars(trainable_vars, parameters):

    l = len(parameters)
    replace = []
    for i in range(l):
        assign = tf.assign(trainable_vars[i], parameters[i])
        replace.append(assign)
    return replace


def separate_grads_and_vars(grads_and_vars_list):

    grads = {}

    i = 0
    k = 1
    for grads_and_vars in grads_and_vars_list:

        j = math.floor(i / 8)
        if k == 1:
            grads["dw" + str(j + 1)] = grads_and_vars[0]
        elif k == 2:
            grads["db" + str(j + 1)] = grads_and_vars[0]
        elif k == 3:
            grads["dgama" + str(j + 1)] = grads_and_vars[0]
        elif k == 4:
            grads["dbeta" + str(j + 1)] = grads_and_vars[0]
        elif k == 5:
            grads["se"+str(j + 1)+"_dw1"] = grads_and_vars[0]
        elif k == 6:
            grads["se"+str(j + 1)+"_db1"] = grads_and_vars[0]
        elif k == 7:
            grads["se"+str(j + 1)+"_dw2"] = grads_and_vars[0]
        elif k == 8:
            k = 0
            grads["se"+str(j + 1)+"_db2"] = grads_and_vars[0]

        i += 1
        k += 1

    return grads


def tcp_connection(ip_address, port):

    workersocket = socket(AF_INET, SOCK_STREAM)
    workersocket.connect((ip_address, port))
    print("Connect Success! Worker ready to receive the initial parameters.")

    return workersocket


def close_socket(workersocket):
    # 关闭套接字，训练结束
    workersocket.send(b'0x03')
    workersocket.close()
    print("Socket closed!")


def recv_initial_parameters(workersocket):
    data = b""
    while True:
        pull_initial_parameters = workersocket.recv(2048000000)
        data += pull_initial_parameters
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    print("Receive the initial parameters success ! Worker start training !")

    return parameters


def push_gradients_to_server(workersocket, grads):
    data = b""
    drumps_grads = pk.dumps(grads)
    workersocket.send(drumps_grads)  # send the grad to server

    while True:
        pull_new_parameters = workersocket.recv(2048000000)
        data += pull_new_parameters
        if len(data) == FLAGS.len:
            break
    parameters = pk.loads(data)
    return parameters


def main():

    workersocket = tcp_connection(FLAGS.ip_address, FLAGS.port)

    X, target_a, target_b, parameters, lam_tensor = create_placeholder()
    # 加入正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)

    logits, regularization, istrain = forward_propagation(X, regularizer=regularizer)

    logits, cost, grads_and_vars = compute_cost(logits, target_a, target_b, lam_tensor, regularization, FLAGS.lr)

    accuracy = compute_accuracy(logits, target_a)

    train_xs, train_labels, test_feeds = load_dataset()

    init = tf.global_variables_initializer()

    trainable_vars = tf.trainable_variables()

    replace = replace_trainable_vars(trainable_vars, parameters)   # recv initial parameters from server

    init_parameters = recv_initial_parameters(workersocket)

    p = convert_dict_to_tuple(init_parameters)

    saver = tf.train.Saver()

    train_accs = []
    test_accs = []
    costs = []
    max_acc = 0
    avg_acc = 0
    prev_acc = 0
    biased_acc = 1
    move_avg_acc = 0
    epoch = 0

    mini_batch_num = int(train_xs.shape[0] / FLAGS.mini_batch_size)

    with tf.Session() as sess:

        sess.run(init)

        while biased_acc > FLAGS.delta_acc:
            epoch_cost = 0
            epoch += 1
            com_time = 0
            mini_batches = random_mini_batches(train_xs, train_labels, FLAGS.mini_batch_size, seed=epoch)
            worker_batches = worker_random_minibatches(mini_batches, worker_minibatch_size=FLAGS.worker_batch_size, seed=epoch+1)
            for worker_batch in worker_batches:
                (worker_batch_X, worker_batch_Y) = worker_batch[FLAGS.partition]
                x, mix_x, y_a, y_b, lam = mixup_data(worker_batch_X, worker_batch_Y, alpha=1)
                _, ops, temp_cost, grads_and_vars_list, train_acc = sess.run([replace, tf.get_collection(tf.GraphKeys.UPDATE_OPS), cost, grads_and_vars, accuracy],
                                                     feed_dict={X: mix_x, target_a: y_a, target_b: y_b, lam_tensor: lam, istrain: True, parameters: p})
                epoch_cost += temp_cost / mini_batch_num
                grads = separate_grads_and_vars(grads_and_vars_list)
                com_start = time.time()
                new_parameters = push_gradients_to_server(workersocket, grads)
                com_end = time.time()
                com_time += (com_end - com_start)
                p = convert_dict_to_tuple(new_parameters)

            costs.append(epoch_cost)
            train_accs.append(train_acc)
            if epoch % 1 == 0:
                print("Epoch {}, Worker{}, Loss = {}, Train_acc = {:.4f}, Communication Time = {:.4f} (s), Biased_acc = {:.5f}".
                      format(epoch, FLAGS.partition+1, epoch_cost, train_acc, com_time, biased_acc))

            for test_feed in test_feeds:
                test_acc = sess.run(accuracy, feed_dict={X: test_feed[0], target_a: test_feed[1], istrain: False})
                avg_acc += test_acc / 10
            test_accs.append(avg_acc)
            if avg_acc > max_acc:
                max_acc = avg_acc
                saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

            if epoch % 5 == 0:
                print("Epoch {}, Worker{}, Avg_acc = {:.4f}, Max_acc = {:.4f}".format(epoch, FLAGS.partition+1, avg_acc, max_acc))

            delta_acc = abs(avg_acc - prev_acc)
            move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
            biased_acc = move_avg_acc / (1 - 0.9**epoch)

            prev_acc = avg_acc
            avg_acc = 0
        # close socket
        close_socket(workersocket)
        # load saved model
        model_file = tf.train.latest_checkpoint("./save_model/")
        saver.restore(sess, model_file)

        print("Loads the saved model: ")
        for test_feed in test_feeds:
            test_acc = sess.run(accuracy, feed_dict={X: test_feed[0], target_a: test_feed[1], istrain: False})
            avg_acc += test_acc / 10
            print("Test accuracy : {:.4f}".format(test_acc))

        print("Average test accuracy is {:.4f}".format(avg_acc))

        with open('./results/test_accs', 'wb') as f:
            f.write(pk.dumps(test_accs))
        with open('./results/loss', 'wb') as f:
            f.write(pk.dumps(costs))


if __name__ == '__main__':
    print('Neural Network Configuration: ')
    print('Learning rate: {}'.format(FLAGS.lr))
    print('Mini_batch_size: {}'.format(FLAGS.mini_batch_size))
    print('Worker_batch_size: {}'.format(FLAGS.worker_batch_size))
    print('Data partition: {}'.format(FLAGS.partition))
    print('The convergence condition: {}'.format(FLAGS.delta_acc))
    print('Number of parameters byte: {}'.format(FLAGS.len))

    print('Network Communication Configuration: ')
    print('The ip address of parameter server: {}'.format(FLAGS.ip_address))
    print('The port of parameter server: {}'.format(FLAGS.port))
    time.sleep(1)
    start = time.time()
    main()
    end = time.time()
    run_time = (end - start)/3600
    print("Run time = {:.2f} (h)".format(run_time))










