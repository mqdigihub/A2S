import tensorflow as tf
import numpy as np

# 参数设置
IMAGE_SIZE = 32

CHANELS_SIZE = 3

OUTPUT_SIZE = 10

# 核大小
CONV_SIZE = 3

# 卷积核数
CONV_DEEP1 = 32
CONV_DEEP2 = 64
CONV_DEEP3 = 128
CONV_DEEP4 = 256

def create_placeholder():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANELS_SIZE], name='input_x')
    y_a = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="input_y_a")
    y_b = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="input_y_b")
    lam_tensor = tf.placeholder(tf.float32, name='lam')

    is_train = tf.placeholder_with_default(False, (), 'is_train')

    # 模型占位符
    init_conv_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CHANELS_SIZE, CONV_DEEP1], name='init_conv_w')
    init_conv_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='init_conv_b')
    init_gama = tf.placeholder(tf.float32, [CONV_DEEP1], name='init_gama')
    init_beta = tf.placeholder(tf.float32, [CONV_DEEP1], name='init_beta')

    conv1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv1_w')
    conv1_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv1_b')
    conv1_gama = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv1_gama')
    conv1_beta = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv1_beta')

    conv2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv2_w')
    conv2_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv2_b')
    conv2_gama = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv2_gama')
    conv2_beta = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv2_beta')

    conv3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv3_w')
    conv3_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv3_b')
    conv3_gama = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv3_gama')
    conv3_beta = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv3_beta')

    conv4_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1], name='conv4_w')
    conv4_b = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv4_b')
    conv4_gama = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv4_gama')
    conv4_beta = tf.placeholder(tf.float32, [CONV_DEEP1], name='conv4_beta')

    shortcut1_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='shortcut1_w')
    shortcut1_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut1_b')
    shortcut1_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut1_gama')
    shortcut1_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut1_beta')
    shortcut2_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='shortcut2_w')
    shortcut2_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut2_b')
    shortcut2_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut2_gama')
    shortcut2_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='shortcut2_beta')
    conv5_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2], name='conv5_w')
    conv5_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv5_b')
    conv5_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv5_gama')
    conv5_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv5_beta')

    conv6_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv6_w')
    conv6_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv6_b')
    conv6_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv6_gama')
    conv6_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv6_beta')

    conv7_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv7_w')
    conv7_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv7_b')
    conv7_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv7_gama')
    conv7_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv7_beta')

    conv8_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2], name='conv8_w')
    conv8_b = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv8_b')
    conv8_gama = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv8_gama')
    conv8_beta = tf.placeholder(tf.float32, [CONV_DEEP2], name='conv8_beta')

    shortcut3_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='shortcut3_w')
    shortcut3_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut3_b')
    shortcut3_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut3_gama')
    shortcut3_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut3_beta')
    shortcut4_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='shortcut4_w')
    shortcut4_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut4_b')
    shortcut4_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut4_gama')
    shortcut4_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='shortcut4_beta')
    conv9_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3], name='conv9_w')
    conv9_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv9_b')
    conv9_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv9_gama')
    conv9_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv9_beta')

    conv10_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv10_w')
    conv10_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv10_b')
    conv10_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv10_gama')
    conv10_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv10_beta')

    conv11_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv11_w')
    conv11_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv11_b')
    conv11_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv11_gama')
    conv11_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv11_beta')

    conv12_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3], name='conv12_w')
    conv12_b = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv12_b')
    conv12_gama = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv12_gama')
    conv12_beta = tf.placeholder(tf.float32, [CONV_DEEP3], name='conv12_beta')

    shortcut5_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='shortcut5_w')
    shortcut5_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut5_b')
    shortcut5_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut5_gama')
    shortcut5_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut5_beta')
    shortcut6_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='shortcut6_w')
    shortcut6_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut6_b')
    shortcut6_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut6_gama')
    shortcut6_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='shortcut6_beta')
    conv13_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4], name='conv13_w')
    conv13_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv13_b')
    conv13_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv13_gama')
    conv13_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv13_beta')

    conv14_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv14_w')
    conv14_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv14_b')
    conv14_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv14_gama')
    conv14_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv14_beta')

    conv15_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv15_w')
    conv15_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv15_b')
    conv15_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv15_gama')
    conv15_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv15_beta')

    conv16_w = tf.placeholder(tf.float32, [CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4], name='conv16_w')
    conv16_b = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv16_b')
    conv16_gama = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv16_gama')
    conv16_beta = tf.placeholder(tf.float32, [CONV_DEEP4], name='conv16_beta')

    fc_w = tf.placeholder(tf.float32, [CONV_DEEP4, OUTPUT_SIZE], name='fc_w')
    fc_b = tf.placeholder(tf.float32, [OUTPUT_SIZE], name='fc_b')

    parameters = (init_conv_w, init_conv_b, init_gama, init_beta,
                  conv1_w, conv1_b, conv1_gama, conv1_beta, conv2_w, conv2_b, conv2_gama, conv2_beta, conv3_w, conv3_b,
                  conv3_gama, conv3_beta, conv4_w, conv4_b, conv4_gama, conv4_beta,
                  shortcut1_w, shortcut1_b, shortcut1_gama, shortcut1_beta, shortcut2_w, shortcut2_b, shortcut2_gama, shortcut2_beta,
                  conv5_w, conv5_b, conv5_gama, conv5_beta, conv6_w, conv6_b, conv6_gama, conv6_beta, conv7_w, conv7_b, conv7_gama, conv7_beta,
                  conv8_w, conv8_b, conv8_gama, conv8_beta,
                  shortcut3_w, shortcut3_b, shortcut3_gama, shortcut3_beta, shortcut4_w, shortcut4_b, shortcut4_gama, shortcut4_beta,
                  conv9_w, conv9_b, conv9_gama, conv9_beta, conv10_w, conv10_b, conv10_gama, conv10_beta, conv11_w, conv11_b, conv11_gama, conv11_beta,
                  conv12_w, conv12_b, conv12_gama, conv12_beta,
                  shortcut5_w, shortcut5_b, shortcut5_gama, shortcut5_beta, shortcut6_w, shortcut6_b, shortcut6_gama, shortcut6_beta,
                  conv13_w, conv13_b, conv13_gama, conv13_beta, conv14_w, conv14_b, conv14_gama, conv14_beta, conv15_w, conv15_b, conv15_gama, conv15_beta,
                  conv16_w, conv16_b, conv16_gama, conv16_beta,
                  fc_w, fc_b)

    return x, y_a, y_b, lam_tensor, is_train, parameters


def initial_parameters():

    np.random.seed(1)

    # init_conv
    w1 = np.random.randn(CONV_SIZE, CONV_SIZE, CHANELS_SIZE, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CHANELS_SIZE)
    b1 = np.zeros((CONV_DEEP1,))
    gama1 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    beta1 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)

    # conv1
    w2 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w3 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w4 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w5 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP1) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)

    b2 = np.zeros((CONV_DEEP1,))
    b3 = np.zeros((CONV_DEEP1,))
    b4 = np.zeros((CONV_DEEP1,))
    b5 = np.zeros((CONV_DEEP1,))

    gama2 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    gama3 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    gama4 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    gama5 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)

    beta2 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    beta3 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    beta4 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)
    beta5 = np.random.randn(CONV_DEEP1) / np.sqrt(CONV_DEEP1)

    # shortcut1
    w6 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w7 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)

    b6 = np.zeros((CONV_DEEP2,))
    b7 = np.zeros((CONV_DEEP2,))

    gama6 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    gama7 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    beta6 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    beta7 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)

    # conv2
    w8 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP1, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP1)
    w9 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w10 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w11 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP2) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)

    b8 = np.zeros((CONV_DEEP2,))
    b9 = np.zeros((CONV_DEEP2,))
    b10 = np.zeros((CONV_DEEP2,))
    b11 = np.zeros((CONV_DEEP2,))

    gama8 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    gama9 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    gama10 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    gama11 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)

    beta8 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    beta9 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    beta10 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)
    beta11 = np.random.randn(CONV_DEEP2) / np.sqrt(CONV_DEEP2)

    # shortcut2
    w12 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w13 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)

    b12 = np.zeros((CONV_DEEP3,))
    b13 = np.zeros((CONV_DEEP3,))

    gama12 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    gama13 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    beta12 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    beta13 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)

    # conv3
    w14 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP2, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP2)
    w15 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w16 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w17 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP3) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)

    b14 = np.zeros((CONV_DEEP3,))
    b15 = np.zeros((CONV_DEEP3,))
    b16 = np.zeros((CONV_DEEP3,))
    b17 = np.zeros((CONV_DEEP3,))

    gama14 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    gama15 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    gama16 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    gama17 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)

    beta14 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    beta15 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    beta16 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)
    beta17 = np.random.randn(CONV_DEEP3) / np.sqrt(CONV_DEEP3)

    # shortcut3
    w18 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w19 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)

    b18 = np.zeros((CONV_DEEP4,))
    b19 = np.zeros((CONV_DEEP4,))

    gama18 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    gama19 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    beta18 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    beta19 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)

    # conv4
    w20 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP3, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP3)
    w21 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w22 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)
    w23 = np.random.randn(CONV_SIZE, CONV_SIZE, CONV_DEEP4, CONV_DEEP4) / np.sqrt(CONV_SIZE * CONV_SIZE * CONV_DEEP4)

    b20 = np.zeros((CONV_DEEP4,))
    b21 = np.zeros((CONV_DEEP4,))
    b22 = np.zeros((CONV_DEEP4,))
    b23 = np.zeros((CONV_DEEP4,))

    gama20 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    gama21 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    gama22 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    gama23 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)

    beta20 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    beta21 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    beta22 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)
    beta23 = np.random.randn(CONV_DEEP4) / np.sqrt(CONV_DEEP4)

    # fc_layer
    w24 = np.random.randn(CONV_DEEP4, OUTPUT_SIZE) / np.sqrt(CONV_DEEP4)
    b24 = np.zeros((OUTPUT_SIZE,))
    initial_parameter = {'w1':w1, 'b1':b1, 'gama1':gama1, 'beta1':beta1, 'w2':w2, 'b2':b2, 'gama2':gama2, 'beta2':beta2,
                         'w3':w3, 'b3':b3, 'gama3':gama3, 'beta3':beta3, 'w4':w4, 'b4':b4, 'gama4':gama4, 'beta4':beta4,
                         'w5':w5, 'b5':b5, 'gama5':gama5, 'beta5':beta5, 'w6':w6, 'b6':b6, 'gama6':gama6, 'beta6':beta6,
                         'w7':w7, 'b7':b7, 'gama7':gama7, 'beta7':beta7, 'w8':w8, 'b8':b8, 'gama8':gama8, 'beta8':beta8,
                         'w9':w9, 'b9':b9, 'gama9':gama9, 'beta9':beta9, 'w10':w10, 'b10':b10, 'gama10':gama10, 'beta10':beta10,
                         'w11':w11, 'b11':b11, 'gama11':gama11, 'beta11':beta11, 'w12':w12, 'b12':b12, 'gama12':gama12, 'beta12':beta12,
                         'w13':w13, 'b13':b13, 'gama13':gama13, 'beta13':beta13, 'w14':w14, 'b14':b14, 'gama14':gama14, 'beta14':beta14,
                         'w15':w15, 'b15':b15, 'gama15':gama15, 'beta15':beta15, 'w16':w16, 'b16':b16, 'gama16':gama16, 'beta16':beta16,
                         'w17':w17, 'b17':b17, 'gama17':gama17, 'beta17':beta17, 'w18':w18, 'b18':b18, 'gama18':gama18, 'beta18':beta18,
                         'w19':w19, 'b19':b19, 'gama19':gama19, 'beta19':beta19, 'w20':w20, 'b20':b20, 'gama20':gama20, 'beta20':beta20,
                         'w21':w21, 'b21':b21, 'gama21':gama21, 'beta21':beta21, 'w22':w22, 'b22':b22, 'gama22':gama22, 'beta22':beta22,
                         'w23':w23, 'b23':b23, 'gama23':gama23, 'beta23':beta23,
                         'w24':w24, 'b24':b24}

    return initial_parameter

import pickle as pk

parameters = initial_parameters()
print(len(pk.dumps(parameters)))




