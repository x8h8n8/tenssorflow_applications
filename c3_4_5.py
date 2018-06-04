# 一个完整的神经网络训练代码过程
import tensorflow as tf

from numpy.random import RandomState

# 定义batch大小
batch_size = 8

# 定义参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1), name="w2")

# 定义输入和测试数据
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
_y = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    _y * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个sess来运行tensorflow程序
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 输出训练前的参数
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 更新参数
        sess.run(train_step, feed_dict={x:X[start:end], _y:Y[start:end]})

        # 每隔多少个iteration输出所有样本数据的交叉熵
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, _y:Y})
            print("After %d training steps, cross entropy on all dataset is %g"%(i, total_cross_entropy))

    # 输出训练后的参数
    print(sess.run(w1))
    print(sess.run(w2))