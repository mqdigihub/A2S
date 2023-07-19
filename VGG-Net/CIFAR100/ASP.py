"""
time: 2022/4/14
author: tmq
code: ASP (Asynchronous Parallel)
"""

import tensorflow as tf
import pickle as pk
import threading
from sgd import SGD_Optimizer
from init_parameters import initialize_parameters
import utils

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.9, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.99, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 50, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 9512427, """Number of gradients byte.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class ASP():

    def __init__(self, init_parameters, lock, len_of_parameters, iteration, epoch):
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
                print('ASP: Min Epoch = {}, Learning Rate = {}'.format(min(self.ep.values()), self.lr))

            # self.lock.acquire()
            self.parameters = SGD_Optimizer(self.lr, self.parameters, grads)
            # self.lock.release()
            socket.send(pk.dumps(self.parameters))

    def counter(self, socket):
        self.t[str(socket)] += 1
        if self.t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def exponential_decay(self):

        self.lr = self.init_lr * self.decay_rate**(min(self.ep.values()))


def main():

    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//8
    lock = threading.Lock()
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:

            t, ep = utils.create_utils(sockets)
            # 实例化
            asp = ASP(init_parameters,
                      lock,
                      len_of_parameters,
                      t,
                      ep)
            for socket in sockets:
                init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                init.start()
                init.join()
                worker_id += 1
            print("Start The Distributed Deep Learning Training! ")
            worker_id = 1
            for socket in sockets:
                asp_thread = threading.Thread(target=asp.async_update, args=(socket, worker_id,))
                asp_thread.setDaemon(True)
                asp_thread.start()
                worker_id += 1


if __name__=='__main__':
    main()







