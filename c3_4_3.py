# tensor, operation 和tensorflow的运行流
import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name="w2")

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# sess.run(w1.initializer)
# sess.run(w1.initializer)
print(sess.run(y))
print(tf.__version__)

sess.close()

