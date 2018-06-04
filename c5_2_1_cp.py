# tensorflow完整的神经网络解决MNIST识别问题
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE =784
OUTPUT_NODE = 10
LAYER1_NODE = 500   #500个影藏节点
BATCH_SIZE = 100    #batch大小
LEARNING_RATE_BASE = 0.1    #基础学习率
TRAINING_STEPS = 10000  #训练迭代轮数

def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name="x-input")
    _y = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name="y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1), name="weight1")
    biases1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]), name="bias1")

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1), name="weight2")
    biases2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]), name="bias2")

    # 前向传播，不使用滑动平均
    layer1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
    y = tf.matmul(layer1, weights2) + biases2

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=_y)
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE)\
        .minimize(loss, global_step=global_step)


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 验证数据集
        validate_feed = {x: mnist.validation.images, _y: mnist.validation.labels}

        # 测试数据集
        test_feed = {x: mnist.test.images, _y:mnist.test.labels}

        # 迭代式训练网络
        for i in range(TRAINING_STEPS):
            # 在每1000伦后输出在验证集上的表现
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using moving average model is: %g"%(i, validate_acc))
                #print(validate_acc)
                #lr = sess.run(learning_rate)
                #print("The learning rate is %g"%lr)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, _y: ys})

        # 训练结束后输出测试集上表现
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("Test accuracy with moving average model is %g"%test_acc)

def main(argv=None):
    mnist = input_data.read_data_sets("F:\MNIST", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()

