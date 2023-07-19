import numpy as np
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 3072
OUTPUT_NODE = 100

# 输入图片的大小
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 100

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 64
CONV1_SIZE = 3

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 128
CONV3_SIZE = 3

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 128
CONV4_SIZE = 3

# 第五层卷积层的尺寸和深度
CONV5_DEEP = 256
CONV5_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV6_DEEP = 256
CONV6_SIZE = 3

# 第7层卷积层的尺寸和深度
CONV7_DEEP = 256
CONV7_SIZE = 3

# 第8层卷积层的尺寸和深度
CONV8_DEEP = 256
CONV8_SIZE = 3

# SEnet权重参数
# SEnet插入第一层卷积层的参数
SE1_SIZE1 = 1
SE1_CHANEL1 = 64
SE1_DEEP1 = 16

SE1_SIZE2 = 1
SE1_CHANEL2 = 16
SE1_DEEP2 = 64

# SEnet插入第二层卷积层的参数
SE2_SIZE1 = 1
SE2_CHANEL1 = 64
SE2_DEEP1 = 16

SE2_SIZE2 = 1
SE2_CHANEL2 = 16
SE2_DEEP2 = 64

# SEnet插入第三层卷积层的参数
SE3_SIZE1 = 1
SE3_CHANEL1 = 128
SE3_DEEP1 = 16

SE3_SIZE2 = 1
SE3_CHANEL2 = 16
SE3_DEEP2 = 128

# SEnet插入第四层卷积层的参数
SE4_SIZE1 = 1
SE4_CHANEL1 = 128
SE4_DEEP1 = 16

SE4_SIZE2 = 1
SE4_CHANEL2 = 16
SE4_DEEP2 = 128

# SEnet插入第五层卷积层的参数
SE5_SIZE1 = 1
SE5_CHANEL1 = 256
SE5_DEEP1 = 16

SE5_SIZE2 = 1
SE5_CHANEL2 = 16
SE5_DEEP2 = 256

# SEnet插入第六层卷积层的参数
SE6_SIZE1 = 1
SE6_CHANEL1 = 256
SE6_DEEP1 = 16

SE6_SIZE2 = 1
SE6_CHANEL2 = 16
SE6_DEEP2 = 256

# SEnet插入第七层卷积层的参数
SE7_SIZE1 = 1
SE7_CHANEL1 = 256
SE7_DEEP1 = 8

SE7_SIZE2 = 1
SE7_CHANEL2 = 8
SE7_DEEP2 = 256

# SEnet插入第八层卷积层的参数
SE8_SIZE1 = 1
SE8_CHANEL1 = 256
SE8_DEEP1 = 8

SE8_SIZE2 = 1
SE8_CHANEL2 = 8
SE8_DEEP2 = 256

# 初始化权重参数


def initialize_parameters():
    np.random.seed(2)

    w1 = np.random.randn(CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP) / np.sqrt(CONV1_SIZE * CONV1_SIZE * NUM_CHANNELS)
    w2 = np.random.randn(CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP) / np.sqrt(CONV2_SIZE * CONV2_SIZE * CONV1_DEEP)
    w3 = np.random.randn(CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP) / np.sqrt(CONV3_SIZE * CONV3_SIZE * CONV2_DEEP)
    w4 = np.random.randn(CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP) / np.sqrt(CONV4_SIZE * CONV4_SIZE * CONV3_DEEP)
    w5 = np.random.randn(CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP) / np.sqrt(CONV5_SIZE * CONV5_SIZE * CONV4_DEEP)
    w6 = np.random.randn(CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP) / np.sqrt(CONV6_SIZE * CONV6_SIZE * CONV5_DEEP)
    w7 = np.random.randn(CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP) / np.sqrt(CONV7_SIZE * CONV7_SIZE * CONV6_DEEP)
    w8 = np.random.randn(CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP) / np.sqrt(CONV8_SIZE * CONV8_SIZE * CONV7_DEEP)
    w9 = np.random.randn(1024, NUM_LABELS) / np.sqrt(1024)

    b1 = np.zeros((CONV1_DEEP,))
    b2 = np.zeros((CONV2_DEEP,))
    b3 = np.zeros((CONV3_DEEP,))
    b4 = np.zeros((CONV4_DEEP,))
    b5 = np.zeros((CONV5_DEEP,))
    b6 = np.zeros((CONV6_DEEP,))
    b7 = np.zeros((CONV7_DEEP,))
    b8 = np.zeros((CONV8_DEEP,))
    b9 = np.zeros((NUM_LABELS,))

    gama1 = np.random.randn(CONV1_DEEP) / np.sqrt(CONV1_DEEP)
    gama2 = np.random.randn(CONV2_DEEP) / np.sqrt(CONV2_DEEP)
    gama3 = np.random.randn(CONV3_DEEP) / np.sqrt(CONV3_DEEP)
    gama4 = np.random.randn(CONV4_DEEP) / np.sqrt(CONV4_DEEP)
    gama5 = np.random.randn(CONV5_DEEP) / np.sqrt(CONV5_DEEP)
    gama6 = np.random.randn(CONV6_DEEP) / np.sqrt(CONV6_DEEP)
    gama7 = np.random.randn(CONV7_DEEP) / np.sqrt(CONV7_DEEP)
    gama8 = np.random.randn(CONV8_DEEP) / np.sqrt(CONV8_DEEP)

    beta1 = np.random.randn(CONV1_DEEP) / np.sqrt(CONV1_DEEP)
    beta2 = np.random.randn(CONV2_DEEP) / np.sqrt(CONV2_DEEP)
    beta3 = np.random.randn(CONV3_DEEP) / np.sqrt(CONV3_DEEP)
    beta4 = np.random.randn(CONV4_DEEP) / np.sqrt(CONV4_DEEP)
    beta5 = np.random.randn(CONV5_DEEP) / np.sqrt(CONV5_DEEP)
    beta6 = np.random.randn(CONV6_DEEP) / np.sqrt(CONV6_DEEP)
    beta7 = np.random.randn(CONV7_DEEP) / np.sqrt(CONV7_DEEP)
    beta8 = np.random.randn(CONV8_DEEP) / np.sqrt(CONV8_DEEP)

    se1_w1 = np.random.randn(SE1_SIZE1, SE1_SIZE1, SE1_CHANEL1, SE1_DEEP1) / np.sqrt(SE1_SIZE1 * SE1_SIZE1 * SE1_CHANEL1)
    se1_w2 = np.random.randn(SE1_SIZE2, SE1_SIZE2, SE1_CHANEL2, SE1_DEEP2) / np.sqrt(SE1_SIZE2 * SE1_SIZE2 * SE1_CHANEL2)
    se2_w1 = np.random.randn(SE2_SIZE1, SE2_SIZE1, SE2_CHANEL1, SE2_DEEP1) / np.sqrt(SE2_SIZE1 * SE2_SIZE1 * SE2_CHANEL1)
    se2_w2 = np.random.randn(SE2_SIZE2, SE2_SIZE2, SE2_CHANEL2, SE2_DEEP2) / np.sqrt(SE2_SIZE2 * SE2_SIZE2 * SE2_CHANEL2)
    se3_w1 = np.random.randn(SE3_SIZE1, SE3_SIZE1, SE3_CHANEL1, SE3_DEEP1) / np.sqrt(SE3_SIZE1 * SE3_SIZE1 * SE3_CHANEL1)
    se3_w2 = np.random.randn(SE3_SIZE2, SE3_SIZE2, SE3_CHANEL2, SE3_DEEP2) / np.sqrt(SE3_SIZE2 * SE3_SIZE2 * SE3_CHANEL2)
    se4_w1 = np.random.randn(SE4_SIZE1, SE4_SIZE1, SE4_CHANEL1, SE4_DEEP1) / np.sqrt(SE4_SIZE1 * SE4_SIZE1 * SE4_CHANEL1)
    se4_w2 = np.random.randn(SE4_SIZE2, SE4_SIZE2, SE4_CHANEL2, SE4_DEEP2) / np.sqrt(SE4_SIZE2 * SE4_SIZE2 * SE4_CHANEL2)
    se5_w1 = np.random.randn(SE5_SIZE1, SE5_SIZE1, SE5_CHANEL1, SE5_DEEP1) / np.sqrt(SE5_SIZE1 * SE5_SIZE1 * SE5_CHANEL1)
    se5_w2 = np.random.randn(SE5_SIZE2, SE5_SIZE2, SE5_CHANEL2, SE5_DEEP2) / np.sqrt(SE5_SIZE2 * SE5_SIZE2 * SE5_CHANEL2)
    se6_w1 = np.random.randn(SE6_SIZE1, SE6_SIZE1, SE6_CHANEL1, SE6_DEEP1) / np.sqrt(SE6_SIZE1 * SE6_SIZE1 * SE6_CHANEL1)
    se6_w2 = np.random.randn(SE6_SIZE2, SE6_SIZE2, SE6_CHANEL2, SE6_DEEP2) / np.sqrt(SE6_SIZE2 * SE6_SIZE2 * SE6_CHANEL2)
    se7_w1 = np.random.randn(SE7_SIZE1, SE7_SIZE1, SE7_CHANEL1, SE7_DEEP1) / np.sqrt(SE7_SIZE1 * SE7_SIZE1 * SE7_CHANEL1)
    se7_w2 = np.random.randn(SE7_SIZE2, SE7_SIZE2, SE7_CHANEL2, SE7_DEEP2) / np.sqrt(SE7_SIZE2 * SE7_SIZE2 * SE7_CHANEL2)
    se8_w1 = np.random.randn(SE8_SIZE1, SE8_SIZE1, SE8_CHANEL1, SE8_DEEP1) / np.sqrt(SE8_SIZE1 * SE8_SIZE1 * SE8_CHANEL1)
    se8_w2 = np.random.randn(SE8_SIZE2, SE8_SIZE2, SE8_CHANEL2, SE8_DEEP2) / np.sqrt(SE8_SIZE2 * SE8_SIZE2 * SE8_CHANEL2)

    se1_b1 = np.zeros((SE1_DEEP1,))
    se1_b2 = np.zeros((SE1_DEEP2,))
    se2_b1 = np.zeros((SE2_DEEP1,))
    se2_b2 = np.zeros((SE2_DEEP2,))
    se3_b1 = np.zeros((SE3_DEEP1,))
    se3_b2 = np.zeros((SE3_DEEP2,))
    se4_b1 = np.zeros((SE4_DEEP1,))
    se4_b2 = np.zeros((SE4_DEEP2,))
    se5_b1 = np.zeros((SE5_DEEP1,))
    se5_b2 = np.zeros((SE5_DEEP2,))
    se6_b1 = np.zeros((SE6_DEEP1,))
    se6_b2 = np.zeros((SE6_DEEP2,))
    se7_b1 = np.zeros((SE7_DEEP1,))
    se7_b2 = np.zeros((SE7_DEEP2,))
    se8_b1 = np.zeros((SE8_DEEP1,))
    se8_b2 = np.zeros((SE8_DEEP2,))

    init_parameters = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6, "w7": w7, "w8": w8, "w9": w9,
                       "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7, "b8": b8, "b9": b9,
                       "gama1": gama1, "gama2": gama2, "gama3": gama3, "gama4": gama4, "gama5": gama5, "gama6": gama6,
                       "gama7": gama7, "gama8": gama8,
                       "beta1": beta1, "beta2": beta2, "beta3": beta3, "beta4": beta4, "beta5": beta5, "beta6": beta6,
                       "beta7": beta7, "beta8": beta8,
                       "se1_w1": se1_w1, "se1_w2": se1_w2, "se2_w1": se2_w1, "se2_w2": se2_w2, "se3_w1": se3_w1, "se3_w2": se3_w2,
                       "se4_w1": se4_w1, "se4_w2": se4_w2, "se5_w1": se5_w1, "se5_w2": se5_w2, "se6_w1": se6_w1, "se6_w2": se6_w2,
                       "se7_w1": se7_w1, "se7_w2": se7_w2, "se8_w1": se8_w1, "se8_w2": se8_w2,
                       "se1_b1": se1_b1, "se1_b2": se1_b2, "se2_b1": se2_b1, "se2_b2": se2_b2, "se3_b1": se3_b1, "se3_b2": se3_b2,
                       "se4_b1": se4_b1, "se4_b2": se4_b2, "se5_b1": se5_b1, "se5_b2": se5_b2, "se6_b1": se6_b1, "se6_b2": se6_b2,
                       "se7_b1": se7_b1, "se7_b2": se7_b2, "se8_b1": se8_b1, "se8_b2": se8_b2
                       }

    return init_parameters


def create_placeholder():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="input_x")
    y_a = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_a")
    y_b = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y_b")
    lam = tf.placeholder(tf.float32, name='lam')


    cv1_w = tf.placeholder(tf.float32, [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], name='cv1_w')
    cv1_b = tf.placeholder(tf.float32, [CONV1_DEEP], name='cv1_b')
    cv1_gama = tf.placeholder(tf.float32, [CONV1_DEEP], name='cv1_gama')
    cv1_beta = tf.placeholder(tf.float32, [CONV1_DEEP], name='cv1_beta')
    se1_w1 = tf.placeholder(tf.float32, [SE1_SIZE1, SE1_SIZE1, SE1_CHANEL1, SE1_DEEP1], name='cv1_se_w1')
    se1_b1 = tf.placeholder(tf.float32, [SE1_DEEP1], name='cv1_se_b1')
    se1_w2 = tf.placeholder(tf.float32, [SE1_SIZE2, SE1_SIZE2, SE1_CHANEL2, SE1_DEEP2], name='cv1_se_w2')
    se1_b2 = tf.placeholder(tf.float32, [SE1_DEEP2], name='cv1_se_b2')

    cv2_w = tf.placeholder(tf.float32, [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], name='cv2_w')
    cv2_b = tf.placeholder(tf.float32, [CONV2_DEEP], name='cv2_b')
    cv2_gama = tf.placeholder(tf.float32, [CONV2_DEEP], name='cv2_gama')
    cv2_beta = tf.placeholder(tf.float32, [CONV2_DEEP], name='cv2_beta')
    se2_w1 = tf.placeholder(tf.float32, [SE2_SIZE1, SE2_SIZE1, SE2_CHANEL1, SE2_DEEP1], name='cv2_se_w1')
    se2_b1 = tf.placeholder(tf.float32, [SE2_DEEP1], name='cv2_se_b1')
    se2_w2 = tf.placeholder(tf.float32, [SE2_SIZE2, SE2_SIZE2, SE2_CHANEL2, SE2_DEEP2], name='cv2_se_w2')
    se2_b2 = tf.placeholder(tf.float32, [SE2_DEEP2], name='cv2_se_b2')

    cv3_w = tf.placeholder(tf.float32, [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], name='cv3_w')
    cv3_b = tf.placeholder(tf.float32, [CONV3_DEEP], name='cv3_b')
    cv3_gama = tf.placeholder(tf.float32, [CONV3_DEEP], name='cv3_gama')
    cv3_beta = tf.placeholder(tf.float32, [CONV3_DEEP], name='cv3_beta')
    se3_w1 = tf.placeholder(tf.float32, [SE3_SIZE1, SE3_SIZE1, SE3_CHANEL1, SE3_DEEP1], name='cv3_se_w1')
    se3_b1 = tf.placeholder(tf.float32, [SE3_DEEP1], name='cv3_se_b1')
    se3_w2 = tf.placeholder(tf.float32, [SE3_SIZE2, SE3_SIZE2, SE3_CHANEL2, SE3_DEEP2], name='cv3_se_w2')
    se3_b2 = tf.placeholder(tf.float32, [SE3_DEEP2], name='cv3_se_b2')

    cv4_w = tf.placeholder(tf.float32, [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP], name='cv4_w')
    cv4_b = tf.placeholder(tf.float32, [CONV4_DEEP], name='cv4_b')
    cv4_gama = tf.placeholder(tf.float32, [CONV4_DEEP], name='cv4_gama')
    cv4_beta = tf.placeholder(tf.float32, [CONV4_DEEP], name='cv4_beta')
    se4_w1 = tf.placeholder(tf.float32, [SE4_SIZE1, SE4_SIZE1, SE4_CHANEL1, SE4_DEEP1], name='cv4_se_w1')
    se4_b1 = tf.placeholder(tf.float32, [SE4_DEEP1], name='cv4_se_b1')
    se4_w2 = tf.placeholder(tf.float32, [SE4_SIZE2, SE4_SIZE2, SE4_CHANEL2, SE4_DEEP2], name='cv4_se_w2')
    se4_b2 = tf.placeholder(tf.float32, [SE4_DEEP2], name='cv4_se_b2')

    cv5_w = tf.placeholder(tf.float32, [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP], name='cv5_w')
    cv5_b = tf.placeholder(tf.float32, [CONV5_DEEP], name='cv5_b')
    cv5_gama = tf.placeholder(tf.float32, [CONV5_DEEP], name='cv5_gama')
    cv5_beta = tf.placeholder(tf.float32, [CONV5_DEEP], name='cv5_beta')
    se5_w1 = tf.placeholder(tf.float32, [SE5_SIZE1, SE5_SIZE1, SE5_CHANEL1, SE5_DEEP1], name='cv5_se_w1')
    se5_b1 = tf.placeholder(tf.float32, [SE5_DEEP1], name='cv5_se_b1')
    se5_w2 = tf.placeholder(tf.float32, [SE5_SIZE2, SE5_SIZE2, SE5_CHANEL2, SE5_DEEP2], name='cv5_se_w2')
    se5_b2 = tf.placeholder(tf.float32, [SE5_DEEP2], name='cv5_se_b2')

    cv6_w = tf.placeholder(tf.float32, [CONV6_SIZE, CONV6_SIZE, CONV5_DEEP, CONV6_DEEP], name='cv6_w')
    cv6_b = tf.placeholder(tf.float32, [CONV6_DEEP], name='cv6_b')
    cv6_gama = tf.placeholder(tf.float32, [CONV6_DEEP], name='cv6_gama')
    cv6_beta = tf.placeholder(tf.float32, [CONV6_DEEP], name='cv6_beta')
    se6_w1 = tf.placeholder(tf.float32, [SE6_SIZE1, SE6_SIZE1, SE6_CHANEL1, SE6_DEEP1], name='cv6_se_w1')
    se6_b1 = tf.placeholder(tf.float32, [SE6_DEEP1], name='cv6_se_b1')
    se6_w2 = tf.placeholder(tf.float32, [SE6_SIZE2, SE6_SIZE2, SE6_CHANEL2, SE6_DEEP2], name='cv6_se_w2')
    se6_b2 = tf.placeholder(tf.float32, [SE6_DEEP2], name='cv6_se_b2')

    cv7_w = tf.placeholder(tf.float32, [CONV7_SIZE, CONV7_SIZE, CONV6_DEEP, CONV7_DEEP], name='cv7_w')
    cv7_b = tf.placeholder(tf.float32, [CONV7_DEEP], name='cv7_b')
    cv7_gama = tf.placeholder(tf.float32, [CONV7_DEEP], name='cv7_gama')
    cv7_beta = tf.placeholder(tf.float32, [CONV7_DEEP], name='cv7_beta')
    se7_w1 = tf.placeholder(tf.float32, [SE7_SIZE1, SE7_SIZE1, SE7_CHANEL1, SE7_DEEP1], name='cv7_se_w1')
    se7_b1 = tf.placeholder(tf.float32, [SE7_DEEP1], name='cv7_se_b1')
    se7_w2 = tf.placeholder(tf.float32, [SE7_SIZE2, SE7_SIZE2, SE7_CHANEL2, SE7_DEEP2], name='cv7_se_w2')
    se7_b2 = tf.placeholder(tf.float32, [SE7_DEEP2], name='cv7_se_b2')

    cv8_w = tf.placeholder(tf.float32, [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP], name='cv8_w')
    cv8_b = tf.placeholder(tf.float32, [CONV8_DEEP], name='cv8_b')
    cv8_gama = tf.placeholder(tf.float32, [CONV8_DEEP], name='cv8_gama')
    cv8_beta = tf.placeholder(tf.float32, [CONV8_DEEP], name='cv8_beta')
    se8_w1 = tf.placeholder(tf.float32, [SE8_SIZE1, SE8_SIZE1, SE8_CHANEL1, SE8_DEEP1], name='cv8_se_w1')
    se8_b1 = tf.placeholder(tf.float32, [SE8_DEEP1], name='cv8_se_b1')
    se8_w2 = tf.placeholder(tf.float32, [SE8_SIZE2, SE8_SIZE2, SE8_CHANEL2, SE8_DEEP2], name='cv8_se_w2')
    se8_b2 = tf.placeholder(tf.float32, [SE8_DEEP2], name='cv8_se_b2')

    fc_w = tf.placeholder(tf.float32, [1024, NUM_LABELS], name='fc_w')
    fc_b = tf.placeholder(tf.float32, [NUM_LABELS], name='fc_b')

    parameters = (
        cv1_w, cv1_b, cv1_gama, cv1_beta, se1_w1, se1_b1, se1_w2, se1_b2,
        cv2_w, cv2_b, cv2_gama, cv2_beta, se2_w1, se2_b1, se2_w2, se2_b2,
        cv3_w, cv3_b, cv3_gama, cv3_beta, se3_w1, se3_b1, se3_w2, se3_b2,
        cv4_w, cv4_b, cv4_gama, cv4_beta, se4_w1, se4_b1, se4_w2, se4_b2,
        cv5_w, cv5_b, cv5_gama, cv5_beta, se5_w1, se5_b1, se5_w2, se5_b2,
        cv6_w, cv6_b, cv6_gama, cv6_beta, se6_w1, se6_b1, se6_w2, se6_b2,
        cv7_w, cv7_b, cv7_gama, cv7_beta, se7_w1, se7_b1, se7_w2, se7_b2,
        cv8_w, cv8_b, cv8_gama, cv8_beta, se8_w1, se8_b1, se8_w2, se8_b2,
        fc_w, fc_b)

    return x, y_a, y_b, parameters, lam
