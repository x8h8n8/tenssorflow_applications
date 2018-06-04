# MNIST数据集使用
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("F:\MNIST", one_hot=True)

print("Training data size: ", mnist.train.num_examples)

print("Validation data size: ", mnist.validation.num_examples)

print("Testing data size: ", mnist.test.num_examples)

print("Example training data: ", mnist.train.images[0])

print("Example training data label: ", mnist.train.labels[0])