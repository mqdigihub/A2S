# import tensorflow as tf
# import sys


# FLAGS = tf.app.flags.FLAGS
# # Network Configuration
# tf.app.flags.DEFINE_integer('batch_size', 3, """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual block per group.
#                                                 Total number of conv layers will be 6n+4""")
#
#
# def main():
#
#     ep = {'a': 3, 'b': 4, 'c': 9}
#     l = sorted(ep.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
#     print(l)
#
#
#
# if __name__=='__main__':
#    main()



# import threading
#
#
# class test():
#
#     def __init__(self):
#
#         self.count = []
#
#     def recv(self, recv_id):
#
#         self.count.append('ok' + str(recv_id))
#         if len(self.count) == 8:
#             print(self.count)
#
#
# def main():
#
#     t = test()
#     for i in range(8):
#         r = threading.Thread(target=t.recv, args=(i,))
#         r.setDaemon(True)
#         r.start()
#     while True: pass
#
# if __name__=='__main__':
#     main()



# a = {'a': 2, 'b': 3, 'c': 4}
# #dict.fromkeys(a, 0)
# print(a)

# 测试
# import numpy as np
# import utils
# from init_parameters import initialize_parameters
#
# init_parameters = initialize_parameters()
# len_of_parameters = len(init_parameters)//2
# grads, m_grads = utils.create_grads_dict(len_of_parameters)
# sockets = [1, 2, 3, 4, 5, 6]
# backup, meansquare = utils.create_backup_w(sockets, init_parameters, grads)
# print(backup)
# print('-----------------')
# print(meansquare)


# import matplotlib.pyplot as plt
#
# result = []
# for i in range(20000):
#
#     decay_lr = 0.085*0.999**i
#     result.append(decay_lr)
#
# plt.plot(result)
# plt.xlabel('Epoch')
# plt.show()
# import pickle as pk
# import struct
#
# # a = [2.0, 3.0, 4.0, 5.0]
# a = {'M': 12.0, 'b': 3.0, 'c': 5.0, 'a': 10}
# print(max(a, key=a.get))
# # a = 0
# pk_a = pk.dumps(a)
# print(len(pk_a))
# pk.loads(pk_a)
# pack_b = struct.pack('f'*4, *a)
# print('--------------------------------------------')
# print(len(pack_b))

# class test():
#
#     def __init__(self, t, A, r_max):
#
#         self.A = A
#         self.t = t
#         self.r_max = r_max
#
#     def record_time(self, socket, t0):
#         self.A[(socket)][1] = self.A[(socket)][0]
#         self.A[(socket)][0] = t0
#
#     def synchronization_controller(self):
#         sim_slowest = []
#         sim_fastest = []
#         fastest_worker = max(self.t, key=self.t.get)
#         slowest_worker = min(self.t, key=self.t.get)
#         interval_f = self.A[str(fastest_worker)][0] - self.A[str(fastest_worker)][1]
#         interval_s = self.A[str(slowest_worker)][0] - self.A[str(slowest_worker)][1]
#         # 模拟r_max次迭代
#         sim_fastest.append(self.A[str(fastest_worker)][0])
#         sim_slowest.append(self.A[str(slowest_worker)][0] + interval_s)
#         for i in range(self.r_max - 1):
#             sim_fastest.append(self.A[str(fastest_worker)][0] + i * interval_f)
#             sim_slowest.append(self.A[str(slowest_worker)][0] + i * interval_s)
#         # 求解最优r
#         print(sim_fastest)
#         print(sim_slowest)
#         i = 0; j = 0; t_min = float('inf'); r_star = 0
#         while i < len(sim_fastest) and j < len(sim_slowest):
#
#             delta_t = abs(sim_fastest[i] - sim_slowest[j])
#             if delta_t < t_min:
#                 t_min = delta_t
#                 r_star = i
#
#             if sim_fastest[i] < sim_slowest[j]:
#                 i += 1
#             else:
#                 j += 1
#
#         return t_min, r_star
#
# def create_time_table(sockets):
#
#     A = {}
#     t0 = 0
#     t1 = 0
#     for i in sockets:
#         A[i] = [t0, t1]
#
#     return A
#
# def main():
#
#     sockets = ['w1', 'w2', 'w3']
#     A = create_time_table(sockets)
#     t = {'w1': 200, 'w2': 100, 'w3': 50}
#     r_max = 12
#
#     t = test(t=t, A=A, r_max=r_max)
#
#     for i in sockets:
#
#         for j in range(2):
#
#             t.record_time(i, time.time())
#             time.sleep(1)
#
#     t_min, r_star = t.synchronization_controller()
#     print(t.A)
#     print(t_min)
#     print(r_star)
#
#
# if __name__=='__main__':
#     main()
#
# import threading
# import time
#
# class test():
#
#     def __init__(self):
#
#         self.t = 0
#
#     def t2(self):
#         while True:
#             time.sleep(1)
#             self.t += 1
#
#     def t1(self):
#
#         self.wait()
#
#     def wait(self):
#
#         while True:
#
#             print(self.t)
#
#
#
# def main():
#
#     ts = test()
#     th1 = threading.Thread(target=ts.t1, args=())
#     th1.setDaemon(True)
#     th1.start()
#
#     th2 = threading.Thread(target=ts.t2, args=())
#     th2.setDaemon(True)
#     th2.start()
#
#     while True:
#         pass
#
#
# if __name__=='__main__':
#     main()

import numpy as np
import math
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("./mnist_data", one_hot=True)


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
        mini_batch = (last_x, last_y)
        mini_batches.append(mini_batch)
    return mini_batches


# reshape_xs = np.reshape(minst.train.images, (minst.train.images.shape[0], 28, 28, 1))
# ys = minst.train.labels  # shape = (55000, 10)
#
# minibatches = random_mini_batches(reshape_xs, ys, mini_batch_size=128, seed=1)
#
# (minibatch_x, minibatch_y) = minibatches[-1]
# minibatch_x = minibatch_x*255
# print(minibatch_x.shape)
# print(minibatch_y.shape)
# print(minibatch_y[131])
# plt.imshow(minibatch_x[131].astype('uint8'))
# plt.show()


# a = np.array([[1, 2, 3],
#               [4, 5, 6]])
#
# b = np.array([[7, 8, 9],
#               [10, 11, 12]])
#
# print(np.r_[a, b])

tuple = [(1, 2)]

b = (3, 4)

tuple[0] = b

print(tuple)