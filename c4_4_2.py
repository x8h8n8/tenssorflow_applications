# 使用tensorflow中的collection进行正则化来预防过拟合
# 计算一个5层的带L2正则化神经网络的损失函数

import tensorflow as tf

def get_weight(shape, _lambda_):
    # 获得神经网络某层的权重，并把他们加入L2正则的集合"losses"中
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(_lambda_)(var))

    return var

x = tf.placeholder(tf.float32, shape=(None,2))
_y = tf.placeholder(tf.float32, shape=(None,1))
batch_size = 8

# 定义每层的节点数
layer_dimension = [2,10,10,10,1]

# 神经网络总层数
n_layers = len(layer_dimension)

# 前向传播当前的层
cur_layer = x
# 当前层节点个数
in_dimension = layer_dimension[0]

# 通过for循环构建5层全链接网络
for i in range(1, n_layers):
    # 定义这一层的输出维度为当前层的节点个数
    out_dimension = layer_dimension[i]
    # 通过get_weight函数获取某一层的权重参数，并加入到集合"losses"中
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 计算前向传播到的当前层的输出，并作为下一层的输入
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 更新下一层的输入维度为这一层的输出维度
    in_dimension = out_dimension

mse_loss = tf.reduce_mean(tf.square(_y - cur_layer))

tf.add_to_collection("losses", mse_loss)

loss = tf.add_n(tf.get_collection("losses"))