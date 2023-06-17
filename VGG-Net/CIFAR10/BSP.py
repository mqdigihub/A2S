"""
time: 2022/4/14
author: tmq
code: BSP (Synchronous Parallel)
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
tf.app.flags.DEFINE_integer('grads_length', 178328, """Number of gradients byte.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class BSP():

    def __init__(self, init_parameters, grads, queue, len_of_parameters):

        self.lr = FLAGS.lr
        self.init_lr = FLAGS.lr
        self.decay_rate = FLAGS.decay_rate
        self.nums_worker = FLAGS.nums_worker
        self.nums_mini_batch = FLAGS.nums_mini_batch
        self.L = FLAGS.grads_length
        self.parameters = init_parameters
        self.queue = queue
        self.l = len_of_parameters
        self.t = 0
        self.ep = 0
        self.grads = grads

    def recv(self, socket, worker_id):
        while True:
            data = b''
            while True:
                pk_grads = socket.recv(2048000000)
                if pk_grads == b'0x03':
                    self.queue['queue1'].put(b'0x03')
                    print('recv {} finished!'.format(worker_id))
                    return  # 结束线程
                data += pk_grads
                if len(data) == self.L:
                    break
            grads = pk.loads(data)
            self.queue['queue1'].put(grads)

    def aggregation(self):

        while True:
            self.grads = dict.fromkeys(self.grads, 0)
            for i in range(self.nums_worker):
                grads = self.queue["queue1"].get()
                if grads == b'0x03':
                    if i == self.nums_worker - 1:
                        for k in range(self.nums_worker):
                            self.queue['queue'+str(k+2)].put(b'0x03')
                        print('aggregation finished!')
                        return
                else:
                    for j in range(self.l + 1):      # 聚合来自各个worker的梯度
                        self.grads['dw' + str(j + 1)] += grads['dw' + str(j + 1)] / self.nums_worker
                        self.grads['db' + str(j + 1)] += grads['db' + str(j + 1)] / self.nums_worker
                        if j != self.l:
                            self.grads["dgama" + str(j + 1)] += grads["dgama" + str(j + 1)] / self.nums_worker
                            self.grads["dbeta" + str(j + 1)] += grads["dbeta" + str(j + 1)] / self.nums_worker
                            self.grads["se" + str(j + 1) + "_dw1"] += grads["se" + str(j + 1) + "_dw1"] / self.nums_worker
                            self.grads["se" + str(j + 1) + "_dw2"] += grads["se" + str(j + 1) + "_dw2"] / self.nums_worker
                            self.grads["se" + str(j + 1) + "_db1"] += grads["se" + str(j + 1) + "_db1"] / self.nums_worker
                            self.grads["se" + str(j + 1) + "_db2"] += grads["se" + str(j + 1) + "_db2"] / self.nums_worker

            self.counter()
            self.parameters = SGD_Optimizer(self.lr, self.parameters, self.grads)  # 更新全局模型参数
            for i in range(self.nums_worker):
                self.queue['queue' + str(i+2)].put(self.parameters)

    def send(self, socket, queue, worker_id):

        while True:
            parameters = queue.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print('send {} finished!'.format(worker_id))
                socket.close()
                return

    def counter(self):
        self.t += 1
        if self.t % self.nums_mini_batch == 0:      # 1个epoch
            self.ep += 1
            self.exponential_decay()
            print('BSP: Epoch = {}, learning rate = {}'.format(self.ep, self.lr))

    def exponential_decay(self):
        self.lr = self.init_lr * self.decay_rate**(self.ep)


def main():

    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//8
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    grads = utils.create_grads_dict(len_of_parameters)
    queue = utils.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            # 实例化
            bsp = BSP(init_parameters,
                      grads,
                      queue,
                      len_of_parameters)
            for socket in sockets:
                bsp_init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                bsp_init.start()
                bsp_init.join()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                bsp_recv = threading.Thread(target=bsp.recv, args=(socket, worker_id,))
                bsp_recv.setDaemon(True)
                bsp_recv.start()
                worker_id += 1

            bsp_agg = threading.Thread(target=bsp.aggregation, args=())
            bsp_agg.setDaemon(True)
            bsp_agg.start()

            worker_id = 1
            for socket in sockets:

                bsp_send = threading.Thread(target=bsp.send, args=(socket, queue['queue'+str(worker_id+1)], worker_id,))
                bsp_send.setDaemon(True)
                bsp_send.start()
                worker_id += 1


if __name__=='__main__':
    main()

