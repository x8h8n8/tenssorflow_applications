import tensorflow as tf

# 参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的size和deep
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的size和deep
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全链接层节点个数
FC_SIZE = 512

# 定义LeNet-5前向传播过程
def inference(input_tensor, train, regularizer):
    # Tips: 使用不同的名字空间来隔离不同层的变量，可以使变量不用担心重命名的问题
    # 采用全0填充，28*28*1 -> 28*28*32
    with tf.variable_scope("layer1-conv1"):
        # 卷积层1
        conv1_weight = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_bias = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1,1,1,1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.name_scope("layer2-pool1"):
        # 第二层池化层， 过滤器边长为2， 全0填充，移动步长是2
        # 28*28*32 -> 14*14*32
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"
        )

    with tf.variable_scope("layer3-conv2"):
        # 第三层卷积层
        # 14*14*32 -> 14*14*64
        conv2_weight = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_bias = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    with tf.name_scope("layer4-pool2"):
        # 第四层池化层
        # 14*14*64 -> 7*7*64
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 将第四层池化层的输出转化为第五层全链接层的输入格式
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope("layer5-fc1"):
        # 第五层全链接层
        # 训练时加入dropout
        fc1_weights = tf.get_variable(
            "weights", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc1_biases = tf.get_variable(
            "biases", [FC_SIZE],
            initializer=tf.constant_initializer(0.1)
        )
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weights))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        # 第六层全链接层
        fc2_weights = tf.get_variable(
            "weights", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc2_biases = tf.get_variable(
            "biases", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )

        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


