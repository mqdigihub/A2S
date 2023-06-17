"""
time: 2022/4/10
author: tmq
code: A2S (Adaptive synchronization strategy)
"""

import tensorflow as tf
import pickle as pk
import threading
from CIFAR100 import utils
from CIFAR100.sgd import SGD_Optimizer
from CIFAR100.init_parameters import initialize_parameters

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.1, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.99, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 55, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 178328, """Number of gradients byte.""")
tf.app.flags.DEFINE_float('beta', 0.9, """The hyper-parameter of move average.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class A2S():

    def __init__(self, lock, init_parameters, len_of_parameters, grads, iteration, epoch, sockets, momentum_grads):
        self.lr = FLAGS.lr
        self.decay_rate = FLAGS.decay_rate
        self.port = FLAGS.port
        self.nums_mini_batch = FLAGS.nums_mini_batch
        self.lock = lock
        self.parameters = init_parameters
        self.L = FLAGS.grads_length
        self.l = len_of_parameters
        self.nums_worker = FLAGS.nums_worker
        self.t = iteration                      # iterations
        self.ep = epoch                     # epoch
        self.count = []
        self.stale_v = 0
        self.sync_group = []
        self.sockets = sockets
        self.init_lr = FLAGS.lr
        self.grads = grads
        self.momentum_grads = momentum_grads
        self.beta = FLAGS.beta
        self.sync_t = 0

    def recv(self, socket, queue1, worker_id):
        while True:
            data = b''
            while True:
                pk_grads = socket.recv(2048000000)
                if pk_grads == b'0x03':
                    self.lock.acquire()
                    queue1.put(socket)
                    queue1.put(b'0x03')
                    self.lock.release()
                    print('recv {} finished!'.format(worker_id))
                    return                  # 结束线程
                data += pk_grads
                if len(data) == self.L:
                    break
            grads = pk.loads(data)
            self.lock.acquire()
            queue1.put(socket)
            queue1.put(grads)
            self.lock.release()

    def adaptive_synchronization(self, queue1):

        socket_and_grads = []
        clock = 0

        while True:
            socket = queue1['queue1'].get()
            grads = queue1['queue1'].get()

            if grads != b'0x03':                    # 结束标志
                self.counter(socket)                # 计数器
                if len(self.count) == self.nums_worker and len(self.sync_group) == 0:
                    self.count.clear()              # 清空
                    self.monitor()                  # 监视器，监视梯度过时情况
                    self.exponential_decay()        # 学习率衰减
                    print('A2S: Min Epoch = {}, Learning Rate = {}, Stale Value = {}'.format(min(self.ep.values()),
                                                                                             self.lr,
                                                                                             self.stale_v))
                # 松弛同步条件，只要同步组中超过1个成员，便开始计数；
                # 超过一定迭代次数则直接聚合无需等待剩余的worker；
                if len(socket_and_grads) > 0:
                    clock += 1
                if str(socket) in self.sync_group:     # 同步组
                    socket_and_grads.append((socket, grads))
                    if len(socket_and_grads) == self.stale_v or clock > 10:
                        clock = 0
                        self.sync_update(socket_and_grads, len(socket_and_grads), queue1)      # 同步更新
                        socket_and_grads.clear()
                        if len(self.count) == self.nums_worker:
                            self.sync_group.clear()    # 清空同步组
                else:                                  # 不是同步组，那就是异步组
                    index = self.sockets.index(socket)
                    self.async_update(grads, queue1['queue' + str(index + 2)])        # 异步更新

            else:
                del self.t[str(socket)]
                del self.ep[str(socket)]
                if socket in self.count:
                    self.count.remove(socket)
                index = self.sockets.index(socket)
                queue1['queue' + str(index+2)].put(b'0x03')
                self.nums_worker = self.nums_worker - 1
                if self.nums_worker == 0:
                    print('A2S Finished!')
                    return
                elif str(socket) in self.sync_group:
                    self.sync_group.remove(str(socket))
                    self.stale_v = self.stale_v - 1

                if len(socket_and_grads) != 0:
                    self.sync_update(socket_and_grads, len(socket_and_grads), queue1)
                    socket_and_grads.clear()
                    self.sync_group.clear()


    def counter(self, socket):
        self.t[str(socket)] += 1
        if self.t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def monitor(self):

        rank = sorted(self.ep.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        self.stale_v = rank[0][1] - rank[self.nums_worker-1][1]
        # stale_v范围：[2, N-1]
        if 1 < self.stale_v < self.nums_worker:
            for i in range(self.stale_v):
                self.sync_group.append(rank[i][0])
        elif self.stale_v >= self.nums_worker:
            for i in range(self.nums_worker - 1):
                self.sync_group.append(rank[i][0])

    def async_update(self, grads, queue1):

        self.parameters = SGD_Optimizer(self.lr, self.parameters, grads)
        queue1.put(self.parameters)

    def sync_update(self, socket_and_grads, s_workers_t, queue1):

        socket = []
        self.grads = dict.fromkeys(self.grads, 0)
        for i in range(len(socket_and_grads)):
            s, g = socket_and_grads[i]
            socket.append(s)
            weight = 1/s_workers_t
            for j in range(self.l):
                self.grads['dw' + str(j + 1)] += g['dw' + str(j + 1)] * weight
                self.grads['db' + str(j + 1)] += g['db' + str(j + 1)] * weight

        self.parameters = SGD_Optimizer(self.lr, self.parameters, self.grads)
        for i in socket:
            index = self.sockets.index(i)
            queue1['queue' + str(index+2)].put(self.parameters)

    def send(self, socket, queue1, worker_id):

        while True:
            parameters = queue1.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print('send {} finished!'.format(worker_id))
                socket.close()
                return

    def exponential_decay(self):

        self.lr = self.init_lr * self.decay_rate**(min(self.ep.values()))

    def add_momentum(self):

        grads = self.grads
        self.sync_t += 1
        for i in range(self.l):
            self.momentum_grads['dw' + str(i + 1)] = self.momentum_grads['dw' + str(i + 1)] * self.beta + \
                                                     (1 - self.beta) * grads['dw' + str(i + 1)]
            self.grads['dw' + str(i + 1)] = self.momentum_grads['dw' + str(i + 1)] / (1 - self.beta ** self.sync_t)

            self.momentum_grads['db' + str(i + 1)] = self.momentum_grads['db' + str(i + 1)] * self.beta + \
                                                     (1 - self.beta) * grads['db' + str(i + 1)]
            self.grads['db' + str(i + 1)] = self.momentum_grads['db' + str(i + 1)] / (1 - self.beta ** self.sync_t)


def main():
    lock = threading.Lock()
    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//2
    queue = utils.create_queue(FLAGS.nums_worker)
    grads, momentum_grads = utils.create_grads_dict(FLAGS.nums_worker)
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        print("Worker:", addr, "Logged on parameter server")
        if len(sockets) == FLAGS.nums_worker:
            t, ep = utils.create_utils(sockets)
            a2s = A2S(lock=lock,
                      init_parameters=init_parameters,
                      len_of_parameters=len_of_parameters,
                      grads=grads,
                      iteration=t,
                      epoch=ep,
                      sockets=sockets,
                      momentum_grads=momentum_grads)
            for socket in sockets:
                a2s_init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                a2s_init.start()
                a2s_init.join()
                worker_id += 1
            print("Start The Distributed Deep Learning Training! ")
            worker_id = 1
            for socket in sockets:

                a2s_recv = threading.Thread(target=a2s.recv, args=(socket, queue['queue1'], worker_id,))
                a2s_recv.setDaemon(True)
                a2s_recv.start()
                worker_id += 1

            a2s_ass = threading.Thread(target=a2s.adaptive_synchronization, args=(queue,))
            a2s_ass.setDaemon(True)
            a2s_ass.start()

            worker_id = 1
            for socket in sockets:

                a2s_send = threading.Thread(target=a2s.send, args=(socket, queue['queue'+str(worker_id+1)], worker_id,))
                a2s_send.setDaemon(True)
                a2s_send.start()
                worker_id += 1


if __name__ == '__main__':
    main()
