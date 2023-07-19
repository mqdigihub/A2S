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

# import time
# import utils
#
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
#     # def synchronization_controller(self):
#     #     #     sim_slowest = []
#     #     #     sim_fastest = []
#     #     #     fastest_worker = max(self.t, key=self.t.get)
#     #     #     slowest_worker = min(self.t, key=self.t.get)
#     #     #     interval_f = self.A[str(fastest_worker)][0] - self.A[str(fastest_worker)][1]
#     #     #     interval_s = self.A[str(slowest_worker)][0] - self.A[str(slowest_worker)][1]
#     #     #     # 模拟r_max次迭代
#     #     #     sim_fastest.append(self.A[str(fastest_worker)][0])
#     #     #     sim_slowest.append(self.A[str(slowest_worker)][0] + interval_s)
#     #     #     for i in range(self.r_max - 1):
#     #     #         sim_fastest.append(self.A[str(fastest_worker)][0] + i * interval_f)
#     #     #         sim_slowest.append(self.A[str(slowest_worker)][0] + i * interval_s)
#     #     #     # 求解最优r
#     #     #     print(sim_fastest)
#     #     #     print(sim_slowest)
#     #     #     i = 0; j = 0; t_min = float('inf'); r_star = 0
#     #     #     while i < len(sim_fastest) and j < len(sim_slowest):
#     #     #
#     #     #         delta_t = abs(sim_fastest[i] - sim_slowest[j])
#     #     #         if delta_t < t_min:
#     #     #             t_min = delta_t
#     #     #             r_star = i
#     #     #
#     #     #         if sim_fastest[i] < sim_slowest[j]:
#     #     #             i += 1
#     #     #         else:
#     #     #             j += 1
#     #     #
#     #     #     return t_min, r_star
#     def synchronization_controller(self):
#         sim_fastest = []
#         sim_slowest = []
#         fastest_worker = max(self.t, key=self.t.get)
#         slowest_worker = min(self.t, key=self.t.get)
#         interval_f = self.A[str(fastest_worker)][0] - self.A[str(fastest_worker)][1]
#         interval_s = self.A[str(slowest_worker)][0] - self.A[str(slowest_worker)][1]
#         # 模拟r_max次迭代
#         sim_fastest.append(self.A[str(fastest_worker)][0])
#         sim_slowest.append(self.A[str(slowest_worker)][0] + interval_s)
#         for i in range(self.r_max - 1):
#
#             sim_fastest.append(self.A[str(fastest_worker)][0] + i*interval_f)
#             sim_slowest.append(self.A[str(slowest_worker)][0] + i*interval_s)
#         # 求解最优r
#         i = 0; j = 0; t_min = float('inf'); r_star = 0
#         while i < len(sim_fastest) and j < len(sim_slowest):
#
#             delta_t = abs(sim_fastest[i] - sim_slowest[j])
#             if delta_t < t_min:
#                 t_min = delta_t
#                 r_star = i
#             if sim_fastest[i] < sim_slowest[j]:
#                 i += 1
#             else:
#                 j += 1
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


# import matplotlib.pyplot as plt
#
# data = [0.111, 0.2111, 0.28122, 0.31123, 0.406666, 0.558, 0.612334, 0.672334, 0.71213123, 0.7523123, 0.75123, 0.771231,
#         0.80123, 0.82213, 0.83123, 0.80123, 0.8412312, 0.82123123, 0.8312312, 0.8512312, 0.8412312, 0.8612312, 0.85123,
#         0.8733, 0.8622, 0.87123, 0.88213, 0.86123, 0.8622, 0.87123, 0.88213, 0.86123, 0.88565, 0.871231, 0.87123, 0.88213,
#         0.86123, 0.88565, 0.89123, 0.89321, 0.8955, 0.8923, 0.8930, 0.8965, 0.8978, 0.9004, 0.8844, 0.8985, 0.8995, 0.8975,
#         0.8985, 0.8983, 0.8975, 0.8973, 0.8972, 0.8982, 0.8970, 0.8975, 0.8977, 0.8978, 0.8980, 0.8976, 0.8977, 0.8978]
#
# acc = 0
# pre_acc = 0
# move_acc = 0
# biased_acc = 0
# b_acc = []
#
# for i in range(len(data)):
#
#     acc = data[i]
#     # delta_acc = abs(acc - pre_acc)
#     move_acc = 0.8*move_acc + 0.2*acc
#     biased_acc = move_acc / (1 - 0.8**(i+1))
#     b_acc.append(biased_acc)
#     print(biased_acc)
#     # pre_acc = acc
#
# plt.plot(b_acc, color='red')
# plt.plot(data, color='green')
# plt.show()


# 二分法
# upper = 8
# lower = 0
# s = 5
# sync_nums = float('inf')
# while sync_nums >= s:
#
#     sync_nums = (upper + lower) // 2
#
#     if sync_nums >= s:
#         upper = sync_nums
#     else:
#         lower = sync_nums
#
# print(sync_nums)


# import threading
#
#
# lock = threading.Lock()
# class test():
#
#     def __init__(self):
#
#         self.s = 0
#
#     def count1(self, id):
#
#         for i in range(50):
#
#             self.s = self.s + 1
#             print('s = {}, id = {}'.format(self.s, id))
#
#
#     def count2(self, id):
#
#         for i in range(50):
#             self.s = self.s - 1
#             print('s = {}, id = {}'.format(self.s, id))
#
# t = test()
#
# a = threading.Thread(target=t.count1, args=(1,))
# b = threading.Thread(target=t.count2, args=(2,))
# a.start()
# b.start()
# a.join()
# b.join()

j = 0
for i in range(1, 11):

    j += i/10

print(j)













