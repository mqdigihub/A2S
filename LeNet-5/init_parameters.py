import numpy as np
import pickle as pk


"""
initialize the parameters
"""


def initialize_parameters():
    np.random.seed(2)

    w1 = np.random.randn(5, 5, 1, 6) / np.sqrt(5 * 5)
    w2 = np.random.randn(5, 5, 6, 16) / np.sqrt(5 * 5 * 6)
    w3 = np.random.randn(256, 120) / np.sqrt(256)
    w4 = np.random.randn(120, 84) / np.sqrt(120)
    w5 = np.random.randn(84, 10) / np.sqrt(84)

    b1 = np.zeros((6,))
    b2 = np.zeros((16,))
    b3 = np.zeros((120,))
    b4 = np.zeros((84,))
    b5 = np.zeros((10,))

    parameters_dict = {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "w5": w5,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "b5": b5
    }

    return parameters_dict