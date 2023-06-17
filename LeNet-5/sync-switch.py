"""
time: 2022/5/2
author: tmq
code: sync-switch (Synchronization Switch)
"""

import tensorflow as tf
import pickle as pk
import threading
from sgd import SGD_Optimizer
from init_parameters import initialize_parameters
import utils

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.0085, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.999, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 55, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 178328, """Number of gradients byte.""")
tf.app.flags.DEFINE_float('switch_time', 0.5, """The upper bound of sync-switch""")
tf.app.flags.DEFINE_integer('total_iterations', 200*55, """Total iterations.""")
# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class Sync_switch():

    def __init__(self, init_parameters, grads, queue, len_of_parameters, lock, t, ep, sockets):

        self.lr = FLAGS.lr
        self.init_lr = FLAGS.lr
        self.decay_rate = FLAGS.decay_rate
        self.nums_worker = FLAGS.nums_worker
        self.nums_mini_batch = FLAGS.nums_mini_batch
        self.L = FLAGS.grads_length
        self.parameters = init_parameters
        self.queue = queue
        self.l = len_of_parameters
        self.grads = grads
        self.total_iterations = FLAGS.total_iterations
        self.synchronization_protocol = 'BSP'
        self.BSP_t = 0
        self.BSP_ep = 0
        self.ASP_t = t
        self.ASP_ep = ep
        self.lock = lock
        self.count = []
        self.switching_timing = 0
        self.sockets = sockets
        self.switching_timing = int(FLAGS.switch_time * self.total_iterations)

    def recv(self, socket, worker_id):
        while True:
            data = b''
            while True:
                pk_grads = socket.recv(2048000000)
                if pk_grads == b'0x03':
                    if socket in self.count:
                        self.count.remove(socket)
                    del self.ASP_t[str(socket)]
                    del self.ASP_ep[str(socket)]
                    self.nums_worker = self.nums_worker - 1
                    print('recv {} finished!'.format(worker_id))
                    return  # 结束线程
                data += pk_grads
                if len(data) == self.L:
                    break
            grads = pk.loads(data)
            if self.synchronization_protocol == 'BSP':
                self.queue['queue1'].put(grads)
            elif self.synchronization_protocol == 'ASP':
                self.asp_counter(socket)
                if len(self.count) == self.nums_worker:
                    self.count.clear()
                    self.exponential_decay(min(self.ASP_ep.values()))
                    print('Sync-Switch: BSP({}%) --> ASP({}%), Current Protocol {}, Min Epoch = {}, Learning Rate = {}'
                          .format(FLAGS.switch_time*100, (1 - FLAGS.switch_time)*100,
                                  self.synchronization_protocol, min(self.ASP_ep.values()), self.lr))

                self.lock.acquire()
                self.parameters = SGD_Optimizer(self.lr, self.parameters, grads)
                self.lock.release()
                socket.send(pk.dumps(self.parameters))

    def aggregation(self):

        while True:
            self.grads = dict.fromkeys(self.grads, 0)
            # BSP聚合
            for i in range(self.nums_worker):
                grads = self.queue["queue1"].get()
                for j in range(self.l):  # 聚合来自各个worker的梯度
                    self.grads["dw" + str(j + 1)] += grads["dw" + str(j + 1)] / self.nums_worker
                    self.grads["db" + str(j + 1)] += grads["db" + str(j + 1)] / self.nums_worker
            self.bsp_counter()
            self.parameters = SGD_Optimizer(self.lr, self.parameters, self.grads)  # 更新全局模型参数
            for i in range(self.nums_worker):
                self.queue['queue' + str(i+2)].put(self.parameters)
            # switch BSP to ASP
            if self.BSP_t >= self.switching_timing:
                self.synchronization_protocol = 'ASP'
                for s in self.sockets:
                    self.ASP_t[str(s)] = self.BSP_t
                    self.ASP_ep[str(s)] = self.BSP_ep
                print('aggregation finished!')
                for i in range(self.nums_worker):
                    self.queue['queue' + str(i + 2)].put(b'0x03')
                return

    def send(self, socket, queue, worker_id):

        while True:
            parameters = queue.get()
            if parameters != b'0x03':
                socket.send(pk.dumps(parameters))
            else:
                print('send {} finished!'.format(worker_id))
                return

    def bsp_counter(self):
        self.BSP_t += 1
        if self.BSP_t % self.nums_mini_batch == 0:      # 1个epoch
            self.BSP_ep += 1
            self.exponential_decay(self.BSP_ep)
            print('Sync-Switch: BSP({}%) --> ASP({}%), Current Protocol: {}, Min Epoch = {}, Learning Rate = {}'
                  .format(FLAGS.switch_time * 100, (1 - FLAGS.switch_time) * 100,
                          self.synchronization_protocol, self.BSP_ep, self.lr))

    def asp_counter(self, socket):
        self.ASP_t[str(socket)] += 1
        if self.ASP_t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ASP_ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def exponential_decay(self, min_epoch):
        self.lr = self.init_lr * self.decay_rate ** min_epoch


def main():

    lock = threading.Lock()
    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//2
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    grads, momentum_grads = utils.create_grads_dict(len_of_parameters)
    queue = utils.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        if len(sockets) == FLAGS.nums_worker:
            t, ep = utils.create_utils(sockets)
            # 实例化
            sync_switch = Sync_switch(init_parameters,
                      grads,
                      queue,
                      len_of_parameters,
                      lock,
                      t,
                      ep,
                      sockets)
            for socket in sockets:
                sync_switch_init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                sync_switch_init.start()
                sync_switch_init.join()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                sync_switch_recv = threading.Thread(target=sync_switch.recv, args=(socket, worker_id,))
                sync_switch_recv.setDaemon(True)
                sync_switch_recv.start()
                worker_id += 1

            sync_switch_agg = threading.Thread(target=sync_switch.aggregation, args=())
            sync_switch_agg.setDaemon(True)
            sync_switch_agg.start()

            worker_id = 1
            for socket in sockets:

                sync_switch_send = threading.Thread(target=sync_switch.send, args=(socket, queue['queue'+str(worker_id+1)], worker_id,))
                sync_switch_send.setDaemon(True)
                sync_switch_send.start()
                worker_id += 1

if __name__=='__main__':
    main()

