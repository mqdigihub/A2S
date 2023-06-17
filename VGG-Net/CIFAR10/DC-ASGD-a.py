"""
time: 2022/4/16
author: tmq
code: DC-ASGD (DC-Asynchronous-SGD)
"""

import tensorflow as tf
import numpy as np
import pickle as pk
import threading
from CIFAR100.sgd import SGD_Optimizer
from CIFAR100.init_parameters import initialize_parameters
from CIFAR100 import utils

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.9, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.99, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 50, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 9512427, """Number of gradients byte.""")
tf.app.flags.DEFINE_float('lam', 2.0, """The hyper-parameter of dc-asgd method""")
tf.app.flags.DEFINE_float('m', 0.95, """The hyper-parameter of dc-asgd method""")
# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class DC_ASGD_a():

    def __init__(self, init_parameters, lock, len_of_parameters, iteration, epoch, backup, mean_square):

        self.lr = FLAGS.lr
        self.init_lr = FLAGS.lr
        self.decay_rate = FLAGS.decay_rate
        self.nums_worker = FLAGS.nums_worker
        self.nums_mini_batch = FLAGS.nums_mini_batch
        self.L = FLAGS.grads_length
        self.parameters = init_parameters
        self.lock = lock
        self.l = len_of_parameters
        self.t = iteration
        self.ep = epoch
        self.count = []
        self.backup = backup
        self.lam = FLAGS.lam
        self.m = FLAGS.m
        self.mean_square = mean_square

    def async_update(self, socket, worker_id):
        while True:
            data = b""
            while True:
                pk_grads = socket.recv(2048000000)  # 梯度一般不会一个包就搞定，通常需要分包，所以需要循环接收梯度信息
                if pk_grads == b'0x03':
                    if socket in self.count:
                        self.count.remove(socket)
                    del self.t[str(socket)]
                    del self.ep[str(socket)]
                    self.nums_worker = self.nums_worker - 1
                    print("recv {} closed!".format(worker_id))
                    return
                data += pk_grads         # 因为收到的是字节型数据，所以将数据包有效负载直接相加
                if len(data) == self.L:  # 当收到的梯度信息等于梯度的大小，意味所有梯度均已收到
                    break
            grads = pk.loads(data)       # 将接收到的梯度序列化信息反序列化
            self.counter(socket)
            if len(self.count) == self.nums_worker:
                self.count.clear()
                self.exponential_decay()
                print('DC-ASGD-a: Min Epoch = {}, Learning Rate = {}'.format(min(self.ep.values()), self.lr))
            dc_grads = self.dc_module(grads, socket)
            self.lock.acquire()
            self.parameters = SGD_Optimizer(self.lr, self.parameters, dc_grads)
            self.backup[str(socket)] = self.parameters
            self.lock.release()
            socket.send(pk.dumps(self.parameters))

    def counter(self, socket):
        self.t[str(socket)] += 1
        if self.t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def exponential_decay(self):

        self.lr = self.init_lr * self.decay_rate**(min(self.ep.values()))

    def dc_module(self, grads, socket):
        dc_grads = {}

        for i in range(self.l + 1):

            self.mean_square[str(socket)]['dw' + str(i+1)] = self.mean_square[str(socket)]['dw' + str(i+1)]*self.m + \
                                                             grads['dw'+str(i+1)]*grads['dw'+str(i+1)]*(1 - self.m)
            lam_dw = self.lam/(np.sqrt(self.mean_square[str(socket)]['dw' + str(i+1)]) + 10**(-7))
            dc_grads['dw' + str(i+1)] = grads['dw' + str(i+1)]\
                                        + lam_dw*grads['dw' + str(i+1)]*grads['dw' + str(i+1)]\
                                        * (self.parameters['w'+str(i+1)] - self.backup[str(socket)]['w'+str(i+1)])

            self.mean_square[str(socket)]['db' + str(i+1)] = self.mean_square[str(socket)]['db' + str(i+1)]*self.m + \
                                                             grads['db'+str(i+1)]*grads['db'+str(i+1)]*(1 - self.m)

            lam_db = self.lam/(np.sqrt(self.mean_square[str(socket)]['db' + str(i+1)]) + 10**(-7))
            dc_grads['db' + str(i+1)] = grads['db' + str(i+1)]\
                                        + lam_db*grads['db' + str(i+1)]*grads['db' + str(i+1)]\
                                        * (self.parameters['b'+str(i+1)] - self.backup[str(socket)]['b'+str(i+1)])

            if i != self.l:
                self.mean_square[str(socket)]['dgama' + str(i + 1)] = self.mean_square[str(socket)]['dgama' + str(i + 1)] * self.m + \
                                                                   grads['dgama' + str(i + 1)] * grads['dgama' + str(i + 1)] * (1 - self.m)
                lam_dgama = self.lam / (np.sqrt(self.mean_square[str(socket)]['dgama' + str(i + 1)]) + 10 ** (-7))
                dc_grads['dgama' + str(i + 1)] = grads['dgama' + str(i + 1)] \
                                              + lam_dgama * grads['dgama' + str(i + 1)] * grads['dgama' + str(i + 1)] \
                                              * (self.parameters['gama' + str(i + 1)] - self.backup[str(socket)]['dgama' + str(i + 1)])

                self.mean_square[str(socket)]['dbeta' + str(i + 1)] = self.mean_square[str(socket)]['dbeta' + str(i + 1)] * self.m + \
                                                                   grads['dbeta' + str(i + 1)] * grads['dbeta' + str(i + 1)] * (1 - self.m)
                lam_dbeta = self.lam / (np.sqrt(self.mean_square[str(socket)]['dbeta' + str(i + 1)]) + 10 ** (-7))
                dc_grads['dbeta' + str(i + 1)] = grads['dbeta' + str(i + 1)] \
                                              + lam_dbeta * grads['dbeta' + str(i + 1)] * grads['dbeta' + str(i + 1)] \
                                              * (self.parameters['beta' + str(i + 1)] - self.backup[str(socket)]['dbeta' + str(i + 1)])

                self.mean_square[str(socket)]['se' + str(i + 1) + '_dw1'] = self.mean_square[str(socket)]['se' + str(i + 1) + '_dw1'] * self.m + \
                                                                   grads['se' + str(i + 1) + '_dw1'] * grads['se' + str(i + 1) + '_dw1'] * (1 - self.m)
                lam_dse1_dw1 = self.lam / (np.sqrt(self.mean_square[str(socket)]['se' + str(i + 1) + '_dw1']) + 10 ** (-7))
                dc_grads['se' + str(i + 1) + '_dw1'] = grads['se' + str(i + 1) + '_dw1'] \
                                              + lam_dse1_dw1 * grads['se' + str(i + 1) + '_dw1'] * grads['se' + str(i + 1) + '_dw1'] \
                                              * (self.parameters['se' + str(i + 1) + '_w1'] - self.backup[str(socket)]['se' + str(i + 1) + '_dw1'])

                self.mean_square[str(socket)]['se' + str(i + 1) + '_db1'] = self.mean_square[str(socket)]['se' + str(i + 1) + '_db1'] * self.m + \
                                                                   grads['se' + str(i + 1) + '_db1'] * grads['se' + str(i + 1) + '_db1'] * (1 - self.m)
                lam_dse1_db1 = self.lam / (np.sqrt(self.mean_square[str(socket)]['se' + str(i + 1) + '_db1']) + 10 ** (-7))
                dc_grads['se' + str(i + 1) + '_db1'] = grads['se' + str(i + 1) + '_db1'] \
                                              + lam_dse1_db1 * grads['se' + str(i + 1) + '_db1'] * grads['se' + str(i + 1) + '_db1'] \
                                              * (self.parameters['se' + str(i + 1) + '_b1'] - self.backup[str(socket)]['se' + str(i + 1) + '_db1'])

                self.mean_square[str(socket)]['se' + str(i + 1) + '_dw2'] = self.mean_square[str(socket)]['se' + str(i + 1) + '_dw2'] * self.m + \
                                                                            grads['se' + str(i + 1) + '_dw2'] * grads['se' + str(i + 1) + '_dw2'] * (1 - self.m)
                lam_dse2_dw2 = self.lam / (np.sqrt(self.mean_square[str(socket)]['se' + str(i + 1) + '_dw2']) + 10 ** (-7))
                dc_grads['se' + str(i + 1) + '_dw2'] = grads['se' + str(i + 1) + '_dw2'] + lam_dse2_dw2 * grads['se' + str(i + 1) + '_dw2'] \
                                                       * grads['se' + str(i + 1) + '_dw2'] \
                                                       * (self.parameters['se' + str(i + 1) + '_w2'] - self.backup[str(socket)]['se' + str(i + 1) + '_dw2'])

                self.mean_square[str(socket)]['se' + str(i + 1) + '_db2'] = self.mean_square[str(socket)]['se' + str(i + 1) + '_db2'] \
                                                                            * self.m + grads['se' + str(i + 1) + '_db2'] * grads['se' + str(i + 1) + '_db2'] * (1 - self.m)
                lam_dse2_db2 = self.lam / (np.sqrt(self.mean_square[str(socket)]['se' + str(i + 1) + '_db2']) + 10 ** (-7))
                dc_grads['se' + str(i + 1) + '_db2'] = grads['se' + str(i + 1) + '_db2'] \
                                                       + lam_dse2_db2 * grads['se' + str(i + 1) + '_db2'] * grads['se' + str(i + 1) + '_db2'] \
                                                       * (self.parameters['se' + str(i + 1) + '_b2'] - self.backup[str(socket)]['se' + str(i + 1) + '_db2'])

        return dc_grads


def main():

    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//8
    lock = threading.Lock()
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    grads = utils.create_grads_dict(len_of_parameters)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:

            t, ep = utils.create_utils(sockets)
            backup, mean_square = utils.create_backup(sockets, init_parameters, grads)
            # 实例化
            dc_asgd = DCASGD(init_parameters,
                      lock,
                      len_of_parameters,
                      t,
                      ep,
                      backup,
                      mean_square)
            for socket in sockets:
                init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                init.start()
                init.join()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                dc_thread = threading.Thread(target=dc_asgd.async_update, args=(socket, worker_id,))
                dc_thread.setDaemon(True)
                dc_thread.start()
                worker_id += 1


if __name__=='__main__':
    main()







