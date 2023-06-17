import tensorflow as tf
import numpy as np
from input_data import Cifar10
from load_data import Cifar
import math
from mixup import mixup_data
from initializer import create_placeholder
import pickle as pk
from socket import *
import time
import ResNet

REGULARIZATION = 0.01
FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.9, """Learning rate.""")
tf.app.flags.DEFINE_integer('mini_batch_size', 1000, """Number of mini batch.""")
tf.app.flags.DEFINE_integer('worker_batch_size', 125, """The number of sample for each worker to train.""")
tf.app.flags.DEFINE_integer('partition', 0, """Data partition.""")
tf.app.flags.DEFINE_float('delta_acc', 0.002, """The convergence condition.""")
tf.app.flags.DEFINE_integer('len', 31362751, """Number of parameters byte.""")
tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual units.""")
tf.app.flags.DEFINE_integer('num_classes', 10, """Number of class.""")
tf.app.flags.DEFINE_integer('k', 2, """The width of filter.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')
tf.app.flags.DEFINE_string('ip_address', '10.1.2.95', '''The ip address of parameter server''')


def convert_dic_to_tuple(parameters_dic):

    dic = parameters_dic
    parameters_tuple = (
         dic['w1'], dic['b1'], dic['gama1'], dic['beta1'], dic['w2'], dic['b2'], dic['gama2'], dic['beta2'],
         dic['w3'], dic['b3'], dic['gama3'], dic['beta3'], dic['w4'], dic['b4'], dic['gama4'], dic['beta4'],
         dic['w5'], dic['b5'], dic['gama5'], dic['beta5'], dic['w6'], dic['b6'], dic['gama6'], dic['beta6'],
         dic['w7'], dic['b7'], dic['gama7'], dic['beta7'], dic['w8'], dic['b8'], dic['gama8'], dic['beta8'],
         dic['w9'], dic['b9'], dic['gama9'], dic['beta9'], dic['w10'], dic['b10'], dic['gama10'], dic['beta10'],
         dic['w11'], dic['b11'], dic['gama11'], dic['beta11'], dic['w12'], dic['b12'], dic['gama12'], dic['beta12'],
         dic['w13'], dic['b13'], dic['gama13'], dic['beta13'], dic['w14'], dic['b14'], dic['gama14'], dic['beta14'],
         dic['w15'], dic['b15'], dic['gama15'], dic['beta15'], dic['w16'], dic['b16'], dic['gama16'], dic['beta16'],
         dic['w17'], dic['b17'], dic['gama17'], dic['beta17'], dic['w18'], dic['b18'], dic['gama18'], dic['beta18'],
         dic['w19'], dic['b19'], dic['gama19'], dic['beta19'], dic['w20'], dic['b20'], dic['gama20'], dic['beta20'],
         dic['w21'], dic['b21'], dic['gama21'], dic['beta21'], dic['w22'], dic['b22'], dic['gama22'], dic['beta22'],
         dic['w23'], dic['b23'], dic['gama23'], dic['beta23'],
         dic['w24'], dic['b24'])

    return parameters_tuple


def separate_grads_and_vars(grads_and_vars_list):

    grads = {}
    i = 0
    k = 1
    for grads_and_vars in grads_and_vars_list:

        j = math.floor(i / 4)
        if k == 1:
            grads["dw" + str(j + 1)] = grads_and_vars[0]
        elif k == 2:
            grads["db" + str(j + 1)] = grads_and_vars[0]
        elif k == 3:
            grads["dgama" + str(j + 1)] = grads_and_vars[0]
        elif k == 4:
            k = 0
            grads["dbeta" + str(j + 1)] = grads_and_vars[0]

        i += 1
        k += 1

    return grads


def replace_trainable_vars(trainable_vars, parameters):

    l = len(parameters)
    replace = []
    for i in range(l):
        assign = tf.assign(trainable_vars[i], parameters[i])
        replace.append(assign)
    return replace


def worker_random_minibatches(minibatch, worker_minibatch_size, seed):
    worker_batches = []
    for batch in minibatch:
        batches = random_mini_batches(batch[0], batch[1], worker_minibatch_size, seed)
        worker_batches.append(batches)

    return worker_batches


def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
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


def loads_data(x, y, is_train):

    # cifar10
    cifar10 = Cifar10(path='./cifar-10-batches-py', one_hot=True)
    cifar10._load_data()

    # cifar100

    train_xs = cifar10.images / 255.0
    train_ys = cifar10.labels

    # 准备测试集
    np.random.seed(1)
    permutation = np.random.permutation(cifar10.test.images.shape[0])
    shuffled_tx = cifar10.test.images[permutation, :, :, :] / 255.0
    shuffled_ty = cifar10.test.labels[permutation, :]

    test_feeds = []
    for k in range(10):

        test_xs = shuffled_tx[k*1000:k*1000 + 1000, :, :, :]
        test_labels = shuffled_ty[k*1000:k*1000 + 1000, :]

        test_feed = {x: test_xs, y: test_labels, is_train: False}

        test_feeds.append(test_feed)

    return train_xs, train_ys, test_feeds


def load_cifar100(x, y, is_train):

    cifar100 = Cifar(mode='cifar100', classes=100, path="./cifar-100-python", one_hot=True)
    cifar100.load_cifar100()

    # 准备训练集
    train_xs = cifar100.train.images / 255.0
    train_labels = cifar100.train.labels

    # 准备测试集
    np.random.seed(1)
    permutation = list(np.random.permutation(10000))
    shuffled_tx = cifar100.test.images[permutation, :, :, :] / 255.0
    shuffled_ty = cifar100.test.labels[permutation, :]

    test_feeds = []
    for k in range(10):
        test_xs = shuffled_tx[k * 1000:k * 1000 + 1000, :, :, :]
        test_labels = shuffled_ty[k * 1000:k * 1000 + 1000, :]

        test_feed = {x: test_xs, y: test_labels, is_train: False}

        test_feeds.append(test_feed)

    return train_xs, train_labels, test_feeds


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


def train():

    worker_socket = tcp_connection(FLAGS.ip_address, FLAGS.port)

    X, y_a, y_b, lam_tensor, is_train, parameters = create_placeholder()

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION)

    hp = ResNet.HParams(batch_size=FLAGS.mini_batch_size,
                        num_classes=FLAGS.num_classes,
                        num_residual_units=FLAGS.num_residual_units,
                        k=FLAGS.k,
                        initial_lr=FLAGS.lr)
    network = ResNet.ResNet18(hp=hp, images=X, labels_a=y_a, labels_b=y_b, lam=lam_tensor, is_train=is_train, regularizer=regularizer)
    network.build_model()
    network.build_train_op()
    network.compute_acc()
    network.startup_bn()

    train_xs, train_ys, test_feeds = loads_data(X, y_a, is_train)

    # saver = tf.train.Saver()

    train_vars = tf.trainable_variables()

    replace = replace_trainable_vars(train_vars, parameters)

    init = tf.global_variables_initializer()

    initial_parameters = recv_initial_parameters(worker_socket)

    p = convert_dic_to_tuple(initial_parameters)

    test_accs = []
    costs = []
    epoch = 0
    prev_acc = 0
    move_avg_acc = 0
    biased_acc = 1
    avg_acc = 0

    with tf.Session() as sess:
        sess.run(init)
        num_mini_batch = int(train_xs.shape[0]/FLAGS.mini_batch_size)
        while biased_acc > FLAGS.delta_acc or epoch < 30:
        # for epoch in range(num_epoches):
            epoch += 1
            epoch_cost = 0
            com_time = 0
            mini_batches = random_mini_batches(train_xs, train_ys, FLAGS.mini_batch_size, epoch)
            worker_batches = worker_random_minibatches(mini_batches, worker_minibatch_size=FLAGS.worker_batch_size, seed=epoch+1)
            for worker_batch in worker_batches:
                (worker_batch_x, worker_batch_y) = worker_batch[FLAGS.partition]
                x, mix_x, mix_y, target_a, target_b, lam = mixup_data(worker_batch_x, worker_batch_y, alpha=1)
                replace_op, bn_op, cost, grads_and_vars, train_acc = sess.run([replace, network.bn, network.loss, network.grads_and_vars, network.acc],
                         feed_dict={X: mix_x, y_a: target_a, y_b: target_b, lam_tensor: lam, is_train: True, parameters: p})

                grads = separate_grads_and_vars(grads_and_vars)
                com_start = time.time()
                new_parameters = push_gradients_to_server(worker_socket, grads)
                com_end = time.time()
                com_time += (com_end - com_start)
                p = convert_dic_to_tuple(new_parameters)
                epoch_cost += cost / num_mini_batch

            costs.append(epoch_cost)

            if epoch % 1 == 0:
                print(
                    "Epoch {}, Worker{}, Loss = {}, Train_acc = {:.4f}, Communication Time = {:.4f} (s), Biased_acc = {:.5f}".
                    format(epoch, FLAGS.partition+1, epoch_cost, train_acc, com_time, biased_acc))

            for test_feed in test_feeds:
                test_acc = sess.run(network.acc, feed_dict=test_feed)
                avg_acc += test_acc / 10
            test_accs.append(avg_acc)
            # if avg_acc > max_acc:
            #     max_acc = avg_acc
            #     saver.save(sess, "./save_model/model.ckpt", global_step=epoch)

            if epoch % 5 == 0:
                print("Epoch {}, Worker{}, Avg_acc = {:.4f}".format(epoch, FLAGS.partition+1, avg_acc))

            delta_acc = abs(avg_acc - prev_acc)
            move_avg_acc = 0.9 * move_avg_acc + 0.1 * delta_acc
            biased_acc = move_avg_acc / (1 - 0.9 ** epoch)

            prev_acc = avg_acc
            avg_acc = 0
        # close socket
        close_socket(worker_socket)
        # load saved model
        # model_file = tf.train.latest_checkpoint("./save_model/")
        # saver.restore(sess, model_file)

        print("The converged model accuracy: ")
        for test_feed in test_feeds:
            test_acc = sess.run(network.acc, feed_dict=test_feed)
            avg_acc += test_acc / 10
            print("Test accuracy : {:.4f}".format(test_acc))

        print("Average test accuracy is {:.4f}".format(avg_acc))

        with open('./results/EXP1/A2S/test_accs', 'wb') as f:
            f.write(pk.dumps(test_accs))
        with open('./results/EXP1/A2S/loss', 'wb') as f:
            f.write(pk.dumps(costs))


def main():

    start = time.time()
    train()
    end = time.time()
    run_time = (end - start) / 3600
    print('Run time: {:.2f}'.format(run_time))


if __name__=='__main__':

    main()





