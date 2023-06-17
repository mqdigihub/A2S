"""
time: 2022/4/24
author: tmq
code: SSP (Dynamic Stale Synchronous Parallel)
"""

import tensorflow as tf
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
tf.app.flags.DEFINE_integer('SL', 12, """The low bound of stale threshold.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class SSP():

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
        self.SL = FLAGS.SL

    def recv(self, socket, worker_id, queue):
        while True:
            data = b""
            while True:
                pk_grads = socket.recv(2048000000)  # 梯度一般不会一个包就搞定，通常需要分包，所以需要循环接收梯度信息
                if pk_grads == b'0x03':
                    queue.put(b'0x03')
                    self.nums_worker = self.nums_worker - 1
                    print("recv {} closed!".format(worker_id))
                    return
                data += pk_grads  # 因为收到的是字节型数据，所以将数据包有效负载直接相加
                if len(data) == self.L:  # 当收到的梯度信息等于梯度的大小，意味所有梯度均已收到
                    break
            grads = pk.loads(data)  # 将接收到的梯度序列化信息反序列化
            queue.put(grads)

    def ssp(self, socket, worker_id, queue):

        while True:
            grads = queue.get()
            if grads == b'0x03':
                del self.t[str(socket)]
                del self.ep[str(socket)]
                if socket in self.count:
                    self.count.remove(socket)
                self.nums_worker = self.nums_worker - 1
                print('ssp {} finished!'.format(worker_id))
                return
            self.counter(socket)
            if len(self.count) == self.nums_worker:
                self.count.clear()
                self.exponential_decay()
                print('SSP: Min Epoch = {}, Learning Rate = {}'.format(min(self.ep.values()), self.lr))

            self.monitor(socket, grads)

    def counter(self, socket):
        self.t[str(socket)] += 1
        if self.t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def monitor(self, socket, grads):
        # 如果worker间的间隔 <= 预设的过时值，则直接更新参数、并发送最新的参数给对应worker
        if self.t[str(socket)] - min(self.t.values()) <= self.SL:
            self.update_and_send(socket, grads)
        # 否则，即worker间的迭代间隔 > 预设的过时值，则强制worker停下来，直到迭代间隔小于该阈值
        else:
            self.wait(socket, grads)

    def wait(self, socket, grads):

        while True:
            # 等待，直到小于预设的过时阈值
            if self.t[str(socket)] - min(self.t.values()) <= self.SL:
                self.update_and_send(socket, grads)
                break

    # 学习率衰减
    def exponential_decay(self):

        self.lr = self.init_lr * self.decay_rate**(min(self.ep.values()))

    # 更新和发送参数
    def update_and_send(self, socket, grads):
        self.lock.acquire()
        self.parameters = SGD_Optimizer(self.lr, self.parameters, grads)
        self.lock.release()
        socket.send(pk.dumps(self.parameters))


def main():

    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//8
    lock = threading.Lock()
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = utils.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:

            t, ep = utils.create_utils(sockets)
            # 实例化
            ssp = SSP(init_parameters,
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
                recv_thread = threading.Thread(target=ssp.recv, args=(socket, worker_id, queue['queue' + str(worker_id)]))
                recv_thread.setDaemon(True)
                recv_thread.start()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                ssp_thread = threading.Thread(target=ssp.ssp, args=(socket, worker_id, queue['queue' + str(worker_id)]))
                ssp_thread.setDaemon(True)
                ssp_thread.start()
                worker_id += 1


if __name__=='__main__':
    main()







