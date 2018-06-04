# tensorflow完整的神经网络解决MNIST识别问题
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE =784
OUTPUT_NODE = 10
LAYER1_NODE = 500   #500个影藏节点
BATCH_SIZE = 100    #batch大小
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #正则化参数
TRAINING_STEPS = 10000  #训练迭代轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2,):
    if avg_class == None:
        # 神经网络前向传播时，没有使用滑动平均
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 计算验证集和测试集上准确率时使用滑动平均处理权重参数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1) + avg_class.average(weights1)))
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(biases2))

def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name="x-input")
    _y = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name="y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1), name="weight1")
    biases1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]), name="bias1")

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1), name="weight2")
    biases2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]), name="bias2")

    # 前向传播，不使用滑动平均
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义average_y, 在验证集和测试集上调用滑动平均模型，验证神经网络的健壮性
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 定义loss function, 并加入L2正则化损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

    # 定义衰减型的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,    # 当前迭代轮数
        mnist.train.num_examples / BATCH_SIZE,   # 跑完所有batch时的迭代次数
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

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
            # 在每1000轮后输出在验证集上的表现
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

