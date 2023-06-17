"""
time: 2022/4/21
author: tmq
code: DSSP (Dynamic Stale Synchronous Parallel)
"""

import tensorflow as tf
import pickle as pk
import threading
from sgd import SGD_Optimizer
from init_parameters import initialize_parameters
import utils
import time

FLAGS = tf.app.flags.FLAGS

# Neural Network Configuration
tf.app.flags.DEFINE_float('lr', 0.085, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('decay_rate', 0.999, """A rate for decaying learning rate .""")
tf.app.flags.DEFINE_integer('nums_worker', 8, """Number of workers.""")
tf.app.flags.DEFINE_integer('nums_mini_batch', 55, """Number of mini batches.""")
tf.app.flags.DEFINE_integer('grads_length', 178328, """Number of gradients byte.""")
tf.app.flags.DEFINE_integer('SL', 3, """The low bound of stale threshold.""")
tf.app.flags.DEFINE_integer('r_max', 12, """The number of max extra iterations.""")

# Network Communication Configuration
tf.app.flags.DEFINE_integer('port', 2222, '''The port of parameter server''')


class DSSP():

    def __init__(self, init_parameters, lock, len_of_parameters, iteration, epoch, A, r_star):

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
        self.r_max = FLAGS.r_max
        self.A = A
        self.r_star = r_star

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
                data += pk_grads         # 因为收到的是字节型数据，所以将数据包有效负载直接相加
                if len(data) == self.L:  # 当收到的梯度信息等于梯度的大小，意味所有梯度均已收到
                    break
            grads = pk.loads(data)       # 将接收到的梯度序列化信息反序列化
            queue.put(grads)

    def dssp(self, socket, worker_id, queue):

        while True:

            grads = queue.get()

            if grads == b'0x03':
                del self.A[str(socket)]
                del self.t[str(socket)]
                del self.ep[str(socket)]
                del self.r_star[str(socket)]
                if socket in self.count:
                    self.count.remove(socket)
                self.nums_worker = self.nums_worker - 1
                print('dssp {} finished!'.format(worker_id))
                return
            self.record_time(socket, time.time())
            self.counter(socket)
            if len(self.count) == self.nums_worker:
                self.count.clear()
                self.exponential_decay()
                print('DSSP: Min Epoch = {}, Learning Rate = {}, r* = {}'.format(min(self.ep.values()), self.lr,
                                                                                 self.r_star[max(self.t, key=self.t.get)]))
            if self.r_star[str(socket)] > 0:           # r*>0则直接更新参数，无需等待
                self.update_and_send(socket, grads)
                self.r_star[str(socket)] = self.r_star[str(socket)] - 1
            else:
                self.monitor(socket, grads)

    def counter(self, socket):
        self.t[str(socket)] += 1
        if self.t[str(socket)] % self.nums_mini_batch == 0:      # 1个epoch
            self.ep[str(socket)] += 1
            if self.count.count(socket) == 0:
                self.count.append(socket)

    def monitor(self, socket, grads):

        if self.t[str(socket)] - min(self.t.values()) <= self.SL:
            self.update_and_send(socket, grads)
        else:
            fastest_worker = max(self.t, key=self.t.get)
            # if tp is fastest worker => if tp - t_slowest<= SL => send ok to worker_p (update)
            if str(socket) == fastest_worker:
                self.synchronization_controller(socket)  # return r_star --> extra iterations for fastest worker
                if self.r_star[str(socket)] > 0:
                    self.update_and_send(socket, grads)
            # 当worker的迭代次数超过SL，则该worker停下来等待直到最慢的worker的push次数满足条件
            self.wait(socket, grads)

    def wait(self, socket, grads):

        while True:
            # 等待
            if self.t[str(socket)] - min(self.t.values()) <= self.SL:
                if self.r_star[str(socket)] == 0:
                    self.update_and_send(socket, grads)
                break

    def record_time(self, socket, t0):

        self.A[str(socket)][1] = self.A[str(socket)][0]
        self.A[str(socket)][0] = t0

    def synchronization_controller(self, socket):
        sim_fastest = []
        sim_slowest = []
        fastest_worker = max(self.t, key=self.t.get)
        slowest_worker = min(self.t, key=self.t.get)
        interval_f = self.A[str(fastest_worker)][0] - self.A[str(fastest_worker)][1]
        interval_s = self.A[str(slowest_worker)][0] - self.A[str(slowest_worker)][1]
        # 模拟r_max次迭代
        sim_fastest.append(self.A[str(fastest_worker)][0])
        sim_slowest.append(self.A[str(slowest_worker)][0] + interval_s)
        for i in range(self.r_max - 1):

            sim_fastest.append(self.A[str(fastest_worker)][0] + i*interval_f)
            sim_slowest.append(self.A[str(slowest_worker)][0] + i*interval_s)
        # 求解最优r
        i = 0; j = 0; t_min = float('inf')
        while i < len(sim_fastest) and j < len(sim_slowest):

            delta_t = abs(sim_fastest[i] - sim_slowest[j])
            if delta_t < t_min:
                t_min = delta_t
                self.r_star[str(socket)] = i
            if sim_fastest[i] < sim_slowest[j]:
                i += 1
            else:
                j += 1

    def exponential_decay(self):

        self.lr = self.init_lr * self.decay_rate**(min(self.ep.values()))

    def update_and_send(self, socket, grads):
        self.lock.acquire()
        self.parameters = SGD_Optimizer(self.lr, self.parameters, grads)
        self.lock.release()
        socket.send(pk.dumps(self.parameters))


def main():

    init_parameters = initialize_parameters()
    len_of_parameters = len(init_parameters)//2
    lock = threading.Lock()
    server_socket = utils.tcp_connection(FLAGS.port, FLAGS.nums_worker)
    queue = utils.create_queue(FLAGS.nums_worker)
    sockets = []
    worker_id = 1
    while True:
        connection_socket, addr = server_socket.accept()
        sockets.append(connection_socket)
        print("Worker:", addr, "Logged on parameter server")
        if len(sockets) == FLAGS.nums_worker:

            t, ep = utils.create_utils(sockets)
            A, r_star = utils.create_time_table(sockets)
            # 实例化
            dssp = DSSP(init_parameters,
                      lock,
                      len_of_parameters,
                      t,
                      ep,
                      A,
                      r_star)
            for socket in sockets:
                init = threading.Thread(target=utils.send_init_parameters, args=(socket, init_parameters, worker_id))
                init.start()
                init.join()
                worker_id += 1
            print("Start The Distributed Deep Learning Training! ")
            worker_id = 1
            for socket in sockets:
                recv_thread = threading.Thread(target=dssp.recv, args=(socket, worker_id, queue['queue' + str(worker_id)]))
                recv_thread.setDaemon(True)
                recv_thread.start()
                worker_id += 1
            worker_id = 1
            for socket in sockets:
                dssp_thread = threading.Thread(target=dssp.dssp, args=(socket, worker_id, queue['queue' + str(worker_id)]))
                dssp_thread.setDaemon(True)
                dssp_thread.start()
                worker_id += 1


if __name__=='__main__':
    main()







