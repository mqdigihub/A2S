from socket import *
from queue import Queue
import pickle as pk


# 建立TCP连接
def tcp_connection(port, nums_worker):

    serversocket = socket(AF_INET, SOCK_STREAM)
    serversocket.bind(('', port))
    serversocket.listen(nums_worker)
    print('The Parameter Server Is Ready: ')

    return serversocket


def send_init_parameters(connectionsocket, parameters, worker_id):

    dumps_parameters = pk.dumps(parameters)
    connectionsocket.send(dumps_parameters)
    print("Send the initial parameters to the worker{} success ! ".format(worker_id))


# 创建队列
def create_queue(nums_worker):
    queue_dict = {}
    for i in range(nums_worker + 1):
        queue_dict["queue" + str(i + 1)] = Queue()
    return queue_dict


def create_utils(sockets):
    t = {}
    ep = {}
    for i in sockets:
        t[str(i)] = 0
        ep[str(i)] = 0
    return t, ep


def create_backup(sockets, parameters, grads):

    backup = {}
    meansquare = {}
    for i in sockets:
        backup[str(i)] = parameters
        meansquare[str(i)] = grads
    return backup, meansquare


def create_time_table(sockets):

    A = {}
    r_star = {}
    t0 = 0
    t1 = 0
    for i in sockets:
        A[str(i)] = [t0, t1]
        r_star[str(i)] = 0

    return A, r_star


def create_grads_dict(len_of_parameters):

    grads = {}
    for i in range(len_of_parameters+1):
        grads['dw' + str(i + 1)] = 0
        grads['db' + str(i + 1)] = 0
        if i != len_of_parameters:
            grads["dgama" + str(i + 1)] = 0
            grads["dbeta" + str(i + 1)] = 0
            grads["se" + str(i + 1) + "_dw1"] = 0
            grads["se" + str(i + 1) + "_dw2"] = 0
            grads["se" + str(i + 1) + "_db1"] = 0
            grads["se" + str(i + 1) + "_db2"] = 0

    return grads



