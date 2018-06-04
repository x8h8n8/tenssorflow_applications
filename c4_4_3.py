# 在tensorflow中使用滑动平均模型来提高模型的健壮性
import tensorflow as tf

# 定义用于计算滑动平均的变量
v1 = tf.Variable(0, dtype=tf.float32)

# 定义网络的迭代轮数
step = tf.Variable(0, trainable=False)

# 定义滑动平均的类
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run([v1, ema.average(v1)]))


    sess.run(tf.assign(v1, 5))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 1000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
