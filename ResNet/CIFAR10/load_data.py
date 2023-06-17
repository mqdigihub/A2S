import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf

class Cifar():

    class test():
        def __init__(self):
            self.images = None
            self.labels = None

    class train():
        def __init__(self):
            self.images = None
            self.labels = None

    def __init__(self, mode, classes, path, one_hot=True):

        self.path = path
        self.one_hot = one_hot
        self.mode = mode
        self.classes = classes

    def load_labels_name(self):
        """使用pickle反序列化labels文件，得到存储内容
            cifar10的label文件为“batches.meta”，cifar100则为“meta”
            反序列化之后得到字典对象，可根据key取出相应内容
        """
        with open(self.path, 'rb') as f:
            obj = pk.load(f)
        return obj


    def load_data_cifar(self, path, mode='cifar100'):
        """ load data and labels information from cifar10 and cifar100
        cifar10 keys(): dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        cifar100 keys(): dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
        """
        with open(path, 'rb') as f:
            dataset = pk.load(f, encoding='bytes')
            if mode == 'cifar10':
                data = dataset[b'data']
                labels = dataset[b'labels']
                img_names = dataset[b'filenames']
            elif mode == 'cifar100':
                data = dataset[b'data']
                labels = dataset[b'fine_labels']
                img_names = dataset[b'filenames']
            else:
                print("mode should be in ['cifar10', 'cifar100']")
                return None, None, None

        return data, labels, img_names


    def load_cifar100(self):

        mode = self.mode
        classes = self.classes
        filename = os.path.join(self.path, 'train')
        print("Loading {}".format(filename))
        train_data, train_labels, train_img_names = self.load_data_cifar(path=filename, mode=mode)
        train_data_4d = train_data.reshape(train_data.shape[0], 3, 32, 32)    #
        train_set_labels = np.array(train_labels).reshape([-1, 1])
        self.train.images = train_data_4d.transpose(0, 2, 3, 1)               #


        filename = os.path.join(self.path, 'test')
        print("Loading {}".format(filename))
        test_data, test_labels, test_img_names = self.load_data_cifar(path=filename, mode=mode)
        test_data_4d = test_data.reshape(test_data.shape[0], 3, 32, 32)
        test_set_labels = np.array(test_labels).reshape([-1, 1])

        self.test.images = test_data_4d.transpose(0, 2, 3, 1)

        if self.one_hot == True:
            self.train.labels = self._one_hot(train_set_labels, classes)
            self.test.labels = self._one_hot(test_set_labels, classes)
        else:
            self.train.labels = train_set_labels
            self.test.labels = test_set_labels

        return self.train.images, self.train.labels, self.test.images, self.test.labels


    def _one_hot(self, labels, num):
        size = labels.shape[0]
        label_one_hot = np.zeros([size, num])
        for i in range(size):
            label_one_hot[i, np.squeeze(labels[i])] = 1
        return label_one_hot

#----------------- 测试 ---------------------

# print('----------------- 测试 ---------------------')
# path = '../../VGG-Net/CIFAR100/cifar-100-python/'
#
# cifar100 = Cifar(mode='cifar100', classes=100, path=path, one_hot=True)
# cifar100.load_cifar100()
#
# print(cifar100.train.images.shape)
# print(cifar100.train.labels.shape)
# print(cifar100.test.images.shape)
# print(cifar100.test.labels.shape)
#
# #plt.imshow(cifar100.train.images[0].astype('uint8'))
# with tf.Session() as sess:
#     new_img = tf.image.adjust_brightness(cifar100.train.images[0], -0.5)/255
#     new_img = tf.clip_by_value(new_img, 0.0, 1.0)
#     new_img = new_img*255
#     img = new_img.eval()
#     img = img.reshape(1, 32, 32, 3)
#     train = np.r_[cifar100.train.images, img]
#     print(train.shape)
#     plt.imshow(train[50000].astype('uint8'))
#     plt.show()




