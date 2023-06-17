"""
time: 2022/4/16
author: tmq
code: DC-ASGD (DC-Asynchronous-SGD)
"""

import tensorflow as tf
import pickle as pk
import threading
from sgd import SGD_Optimizer
from initializer import initial_parameters
import utils

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.9, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.99, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 50, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 15684552, """Number of gradients byte.""")
tf.app.flags.DEFINE_float('lam', 0.04, """The hyper-parameter of dc-asgd method""")
# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class DC_ASGD_c():

    def __init__(self, init_parameters, lock, len_of_parameters, iteration, epoch, backup):

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
                print('DC-ASGD-c: Min Epoch = {}, Learning Rate = {}'.format(min(self.ep.values()), self.lr))
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

            dc_grads['dw' + str(i+1)] = grads['dw' + str(i+1)]\
                                        + self.lam*grads['dw' + str(i+1)]*grads['dw' + str(i+1)]\
                                        * (self.parameters['w'+str(i+1)] - self.backup[str(socket)]['w'+str(i+1)])

            dc_grads['db' + str(i+1)] = grads['db' + str(i+1)]\
                                        + self.lam*grads['db' + str(i+1)]*grads['db' + str(i+1)]\
                                        * (self.parameters['b'+str(i+1)] - self.backup[str(socket)]['b'+str(i+1)])

            if i != self.l:
                dc_grads['dgama' + str(i + 1)] = grads['dgama' + str(i + 1)] \
                                              + self.lam * grads['dgama' + str(i + 1)] * grads['dgama' + str(i + 1)] \
                                              * (self.parameters['gama' + str(i + 1)] - self.backup[str(socket)]['dgama' + str(i + 1)])
                dc_grads['dbeta' + str(i + 1)] = grads['dbeta' + str(i + 1)] \
                                                 + self.lam * grads['dbeta' + str(i + 1)] * grads['dbeta' + str(i + 1)] \
                                                 * (self.parameters['beta' + str(i + 1)] - self.backup[str(socket)]['dbeta' + str(i + 1)])

        return dc_grads


def main():

    init_parameters = initial_parameters()
    len_of_parameters = len(init_parameters)//4
    lock = threading.Lock()
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    grads, momentum_grads = utils.create_grads_dict(len_of_parameters)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:

            t, ep = utils.create_utils(sockets)
            backup, mean_square = utils.create_backup(sockets, init_parameters, grads)
            # 实例化
            dc_asgd_c = DC_ASGD_c(init_parameters,
                      lock,
                      len_of_parameters,
                      t,
                      ep,
                      backup)
            for socket in sockets:
                init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                init.start()
                init.join()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                dc_thread = threading.Thread(target=dc_asgd_c.async_update, args=(socket, worker_id,))
                dc_thread.setDaemon(True)
                dc_thread.start()
                worker_id += 1


if __name__=='__main__':
    main()







