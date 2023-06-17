import tensorflow as tf
import numpy as np
import math
from socket import *
import pickle as pk
import time
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("./mnist_data", one_hot=True)

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

BATCH_SIZE = 512

REGULARIZERATION = 0.0001

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 6
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 5

# 第一层全连接层的节点个数
FC1_SIZE = 120

# 第二层全连接层的节点个数
FC2_SIZE = 84

# 第二层池化层拉直后的节点数
POOL2_FLATTEN = 256

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.085, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('delta_acc', 0.0001, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 356022, """Number of parameters byte.""")
tf.app.flags.DEFINE_string('save_acc', './lr_0.085(0.999decay)/A2S_RESULTS/test_accs', """The path of acc.""")
tf.app.flags.DEFINE_string('save_cost', './lr_0.085(0.999decay)/A2S_RESULTS/loss', """The path of cost.""")
# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '10.1.2.95', '''The ip address of parameter server''')


def dataset_process(X, Y, minst=minst, dropout=0):
    # 准备训练数据
    # shape = (55000, 784)
    reshape_xs = np.reshape(minst.train.images, (minst.train.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    ys = minst.train.labels  # shape = (55000, 10)

    # 准备验证数据
    # shape = (5000, 784)
    reshape_validate_xs = np.reshape(minst.validation.images,
                                     (minst.validation.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    validate_ys = minst.validation.labels # shape = (5000, 10)

    validate_feed = {X: reshape_validate_xs, Y: validate_ys, dropout: 0}

    # 准备测试数据
    # shape = (10000, 784)
    reshape_test_xs = np.reshape(minst.test.images, (minst.test.images.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    test_ys = minst.test.labels # shape = (10000, 10)
    test_feed = {X: reshape_test_xs, Y: test_ys, dropout: 0}

    return reshape_xs, ys, validate_feed, test_feed


def create_placeholder():
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x_input')
    Y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_label')

    conv1_weights = tf.placeholder(tf.float32, [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], name='conv1_weight')
    conv1_biases = tf.placeholder(tf.float32, [CONV1_DEEP], name='conv1_biases')

    conv2_weights = tf.placeholder(tf.float32, [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], name='conv2_weight')
    conv2_biases = tf.placeholder(tf.float32, [CONV2_DEEP], name='conv2_biases')

    fc1_weights = tf.placeholder(tf.float32, [POOL2_FLATTEN, FC1_SIZE], name='fc1_weight')
    fc1_biases = tf.placeholder(tf.float32, [FC1_SIZE], name='fc1_biases')

    fc2_weights = tf.placeholder(tf.float32, [FC1_SIZE, FC2_SIZE], name='fc2_weight')
    fc2_biases = tf.placeholder(tf.float32, [FC2_SIZE], name='fc2_biases')

    fc3_weights = tf.placeholder(tf.float32, [FC2_SIZE, NUM_LABELS], name='fc3_weight')
    fc3_biases = tf.placeholder(tf.float32, [NUM_LABELS], name='fc3_biases')

    dropout = tf.placeholder_with_default(1, (), name='dropout')

    return X, Y, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, dropout


def forward_propagation(input_tensor, dropout, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    P = tf.contrib.layers.flatten(pool2)

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [P.shape[1], FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w1 = regularizer(fc1_weights)

        fc1_biases = tf.get_variable('bias', [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(P, fc1_weights) + fc1_biases)
        if dropout == 1: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w2 = regularizer(fc2_weights)
        fc2_biases = tf.get_variable('biases', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if dropout == 1: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            regularization_w3 = regularizer(fc3_weights)
        fc3_biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logits, (regularization_w1 + regularization_w2 + regularization_w3)


def compute_cost(logits, label, learning_rate, regularization):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 加入正则化
    loss = cross_entropy_mean + regularization

    optimizer = tf.train.AdamOptimizer(learning_rate)

    grads_and_vars = optimizer.compute_gradients(loss)

    return logits, loss, optimizer, grads_and_vars


def compute_accuracy(logits, label):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch &lt; mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        (last_x, last_y) = mini_batches[-1]
        last_x = np.r_[last_x, mini_batch_X]
        last_y = np.r_[last_y, mini_batch_Y]
        mini_batches[-1] = (last_x, last_y)
        # mini_batch = (mini_batch_X, mini_batch_Y)
        # mini_batches.append(mini_batch)
    return mini_batches


def worker_random_minibatches(minibatches, worker_minibatch_size, seed=1):
    worker_batches = []
    for batch in minibatches:
        batches = random_mini_batches(batch[0], batch[1], worker_minibatch_size, seed)
        worker_batches.append(batches)

    return worker_batches


def separate_grads_and_vars(grad_and_var_list):
    grads = {}
    i = 0
    for grads_and_vars in grad_and_var_list:

        j = math.floor(i / 2)
        if i % 2 == 0:
            grads["dw" + str(j + 1)] = grads_and_vars[0]
        else:
            grads["db" + str(j + 1)] = grads_and_vars[0]
        i = i + 1
    return grads


def tcp_connection(ip_address, port):
    worker1socket = socket(AF_INET, SOCK_STREAM)  # TCP
    worker1socket.connect((ip_address, port))
    print("Connect Success! Worker ready to receive the initial parameters.")

    return worker1socket


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

    X, Y, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, \
    fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, dropout = create_placeholder()

    # 加入正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERATION)
    logits, regularization = forward_propagation(X, dropout, regularizer=regularizer)

    logit, loss, optimizer, grad_and_var = compute_cost(logits, Y, FLAGS.lr, regularization)

    accuracy = compute_accuracy(logit, Y)

    init = tf.global_variables_initializer()
    trained_vars = tf.trainable_variables()    # 老的模型参数

    # 用来自服务器的模型参数来更新本地的模型参数
    c1_w = tf.assign(trained_vars[0], conv1_weights)
    c1_b = tf.assign(trained_vars[1], conv1_biases)
    c2_w = tf.assign(trained_vars[2], conv2_weights)
    c2_b = tf.assign(trained_vars[3], conv2_biases)
    f1_w = tf.assign(trained_vars[4], fc1_weights)
    f1_b = tf.assign(trained_vars[5], fc1_biases)
    f2_w = tf.assign(trained_vars[6], fc2_weights)
    f2_b = tf.assign(trained_vars[7], fc2_biases)
    f3_w = tf.assign(trained_vars[8], fc3_weights)
    f3_b = tf.assign(trained_vars[9], fc3_biases)

    paras = recv_initial_parameters(workersocket)
    saver = tf.train.Saver()
    Cost = []
    Validation_acc = []
    test_accs = []
    max_acc = 0
    prev_acc = 0
    move_acc = 0
    biased_acc = 1
    epoch = 0
    reshape_xs, ys, validate_feed, test_feed = dataset_process(X, Y, minst, dropout)

    num_mini_batch = int(reshape_xs.shape[0] / FLAGS.mini_batch_size)

    with tf.Session() as sess:
        sess.run(init)

        # while biased_acc > FLAGS.delta_acc:
        for epoch in range(200):
            # epoch += 1
            epoch_cost = 0
            com_time = 0

            mini_batches = random_mini_batches(reshape_xs, ys, FLAGS.mini_batch_size, seed=epoch)
            worker_batches = worker_random_minibatches(mini_batches, FLAGS.worker_batch_size, seed=epoch+1)

            for mini_batch in worker_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch[FLAGS.partition]    # 分配worker训练的数据集分区
                c1_w_, c1_b_, c2_w_, c2_b_, f1_w_, f1_b_, f2_w_, f2_b_, f3_w_, f3_b_, cost, grads_and_vars, train_acc = \
                    sess.run([c1_w, c1_b, c2_w, c2_b, f1_w, f1_b, f2_w, f2_b, f3_w, f3_b, loss, grad_and_var, accuracy],
                             feed_dict={X: mini_batch_X, Y: mini_batch_Y, conv1_weights: paras["w1"],
                                        conv1_biases: paras["b1"], conv2_weights: paras["w2"],
                                        conv2_biases: paras["b2"], fc1_weights: paras["w3"],
                                        fc1_biases: paras["b3"], fc2_weights: paras["w4"],
                                        fc2_biases: paras["b4"], fc3_weights: paras["w5"],
                                        fc3_biases: paras["b5"], dropout: 1})

                grads = separate_grads_and_vars(grads_and_vars)
                com_start = time.time()
                paras = push_gradients_to_server(workersocket, grads)
                com_end = time.time()
                com_time += (com_end - com_start)
                epoch_cost += cost / (num_mini_batch)

            Cost.append(epoch_cost)
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            Validation_acc.append(validate_acc)
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            test_accs.append(test_acc)

            # delta_acc = abs(test_acc - prev_acc)
            # move_acc = 0.9 * move_acc + 0.1 * delta_acc
            # biased_acc = move_acc / (1 - 0.9 ** epoch)

            # prev_acc = test_acc

            # if test_acc > max_acc:
            #     max_acc = test_acc
            #     saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

            if epoch % 1 == 0:
                print("After {} training epochs, training cost is {}, communication time is {:.4f} (sec), Biased acc is {}".format(epoch, epoch_cost,
                                                                                                                                   com_time, biased_acc))

            # 每 5 轮输出一次在验证数据集上的测试结果
            if epoch % 5 == 0:
                print("After {} training epochs, validation accuracy is {:.5f}, test accuracy is {:.5f}".format(epoch, validate_acc, test_acc))

        # model_file = tf.train.latest_checkpoint("./save_model/")
        # saver.restore(sess, model_file)
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("Test accuracy is {}".format(test_acc))

    # 关闭套接字，训练结束
    workersocket.send(b'0x03')
    workersocket.close()
    print("Socket closed!")

    # 将准确度保存为文件
    # with open(FLAGS.save_acc, 'wb') as f:
    #     f.write(pk.dumps(test_accs))
    # with open(FLAGS.save_cost, 'wb') as f:
    #     f.write(pk.dumps(Cost))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    run_time = (end - start)/3600
    print("Run time is {} (h)".format(run_time))

