import tensorflow as tf

def SE_block(x,ratio):

    shape = x.get_shape().as_list()
    channel_out = shape[3]
    # print(shape)
    # 第一层，全局平均池化层
    with tf.variable_scope("se_squeeze"):
        squeeze = tf.nn.avg_pool(x, [1,shape[1],shape[2],1],[1,shape[1],shape[2],1], padding = "SAME")
    # 第二层，全连接层
    with tf.variable_scope("se_fc1"):
        w_excitation1 = tf.get_variable('weight', shape=[1, 1, channel_out, int(channel_out/ratio)], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_excitation1 = tf.get_variable('biase', shape=[int(channel_out/ratio)], initializer=tf.constant_initializer(0.1))
        excitation1 = tf.nn.conv2d(squeeze, w_excitation1, strides=[1, 1, 1, 1], padding="SAME")
        excitation1_output = tf.nn.relu(tf.nn.bias_add(excitation1, b_excitation1))
    # 第三层，全连接层
    with tf.variable_scope("se_fc2"):
        w_excitation2 = tf.get_variable('weight', shape=[1, 1, int(channel_out / ratio), channel_out], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_excitation2 = tf.get_variable('biase', shape=[channel_out], initializer=tf.constant_initializer(0.1))
        excitation2 = tf.nn.conv2d(excitation1_output, w_excitation2, strides=[1, 1, 1, 1], padding="SAME")
        excitation2_output = tf.nn.sigmoid(tf.nn.bias_add(excitation2, b_excitation2))
    # 第四层，点乘
    excitation_output = tf.reshape(excitation2_output,[-1, 1, 1, channel_out])
    h_output = excitation_output * x

    return h_output