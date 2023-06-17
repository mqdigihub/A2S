from collections import namedtuple
import tensorflow as tf
import resnet_utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, '
                    'initial_lr')


class ResNet18(object):
    def __init__(self, hp, images, labels_a, labels_b, lam, is_train, regularizer):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels_a = labels_a
        self._labels_b = labels_b
        self._lam = lam
        self.is_train = is_train
        self.regularizer = regularizer

    def build_model(self):
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = resnet_utils._conv(self._images, 3, 16*self._hp.k, 1, name='init_conv')
        x = resnet_utils._bn(x, self.is_train, name='init_bn')
        x = resnet_utils._relu(x)
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        # Residual Blocks
        filters = [16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k, 128 * self._hp.k]
        strides = [1, 2, 2, 2]

        for i in range(1, 5):
            # First residual unit
            with tf.variable_scope('unit_%d_0' % i) as scope:
                print('\tBuilding residual unit: %s' % scope.name)
                if i == 1:
                    shortcut = x
                else:
                    shortcut = resnet_utils._conv(x, 3, filters[i-1], 1, name='shortcut_c1')
                    shortcut = resnet_utils._bn(shortcut, self.is_train, name='shortcut_bn1')
                    shortcut = resnet_utils._conv(shortcut, 3, filters[i-1], strides[i-1], name='shortcut_c2')
                    shortcut = resnet_utils._bn(shortcut, self.is_train, name='shortcut_bn2')
                # Residual
                x = resnet_utils._conv(x, 3, filters[i-1], strides[i-1], name='conv_1')
                x = resnet_utils._bn(x, self.is_train, name='bn_1')
                x = resnet_utils._relu(x)
                x = resnet_utils._conv(x, 3, filters[i-1], 1, name='conv_2')
                x = resnet_utils._bn(x, self.is_train, name='bn_2')
                # Merge
                x = x + shortcut
                x = resnet_utils._relu(x)

            # Other residual units
            for j in range(1, self._hp.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    # Shortcut
                    shortcut = x
                    # Residual
                    x = resnet_utils._conv(x, 3, filters[i-1], 1, name='conv_1')
                    x = resnet_utils._bn(x, self.is_train, name='bn_1')
                    x = resnet_utils._relu(x)
                    x = resnet_utils._conv(x, 3, filters[i-1], 1, name='conv_2')
                    x = resnet_utils._bn(x, self.is_train, name='bn_2')
                    # Merge
                    x = x + shortcut
                    x = resnet_utils._relu(x)

        # Last unit
        with tf.variable_scope('avg_pool_layer') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            print(x.shape)

        # Logit
        with tf.variable_scope('full_connected_layer') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.contrib.layers.flatten(x)
            x, regurization = resnet_utils._fc(x, self._hp.num_classes, self.regularizer)

        self._logits = x
        self._regurization = regurization

        # Probs & preds & acc

    def build_train_op(self):

        cross_entropy_a = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits,
                                                                         labels=tf.argmax(self._labels_a, 1))
        cross_entropy_b = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits,
                                                                         labels=tf.argmax(self._labels_b, 1))
        cross_entropy_mean_a = tf.reduce_mean(cross_entropy_a)
        cross_entropy_mean_b = tf.reduce_mean(cross_entropy_b)
        self.loss = cross_entropy_mean_a * self._lam + cross_entropy_mean_b * (1 - self._lam) + self._regurization
        self.opt = tf.train.AdamOptimizer(self._hp.initial_lr)
        self.grads_and_vars = self.opt.compute_gradients(self.loss)

    def compute_acc(self):

        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels_a, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def startup_bn(self):

        self.bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)









