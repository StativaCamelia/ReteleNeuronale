import random

import numpy as np


def softmax_activation(z):
	e_z = np.exp(z)
	return e_z / np.sum(e_z, axis=0)


def sigmoid_activation(z):
	return 1 / (1 + np.exp(-z))


class NeuralNetwork:
	def __init__(self, train_set, test_set, label_encode=False):
		self.label_encode = label_encode
		self.no_of_epochs = 80
		self.test_set = test_set
		self.train_set = train_set
		self.test_set_input, self.test_set_label = test_set[0], test_set[1]
		self.train_set_input, self.train_set_label = train_set[0], train_set[1]
		self.no_data_samples = len(self.train_set_input)
		self.len_input = len(self.train_set_input[0])
		self.len_hidden = 100
		self.len_output = 10
		self.mini_batch_len = 1024
		self.learning_rate = 0.03
		self.weights = []
		self.biases = []
		self.layers_sizes = [self.len_input, self.len_hidden, self.len_output]
		self.initialize_biases()
		self.initialize_weights()

	def get_one_hot_vector(self, labels):
		res = np.eye(10)[np.array(labels).reshape(-1)]
		return res.reshape(list(labels.shape) + [10])

	def convert_in_one_hot_vector(self, labels):
		self.train_set_label = self.get_one_hot_vector(labels)


	def initialize_weights(self):
		self.weights = [
			np.random.randn(self.layers_sizes[i], self.layers_sizes[i - 1]) / np.sqrt(self.layers_sizes[i - 1]) for i in
			range(1, len(self.layers_sizes))]

	def initialize_biases(self):
		self.biases = [np.random.randn(self.layers_sizes[i], 1) for i in range(1, len(self.layers_sizes))]

	def feed_forward(self, x):
		activations = []
		net_inputs = []
		activation_pred = x.T.reshape(784, 1)
		activations.append(activation_pred)
		for i in range(len(self.layers_sizes) - 2):
			net_input = np.dot(self.weights[i], activation_pred) + self.biases[i]
			net_inputs.append(net_input)
			activation_pred = sigmoid_activation(net_input)
			activations.append(activation_pred)
		net_input_last = np.dot(self.weights[-1], activation_pred) + self.biases[-1]
		net_inputs.append(net_input_last)
		activation_last = softmax_activation(net_input_last)
		activations.append(activation_last)
		return net_inputs, activations, activation_last


	def cross_entropy(self, activation_last):
		n = self.train_set_label.shape[0]
		return (-1 / n) * np.sum(
			np.multiply(self.train_set_label, np.log(activation_last)) + np.multiply(1 - self.train_set_label,
																					 np.log(1 - activation_last)))

	def cross_entropy_derivative(self, output, target):
		return output.T - target


	def backward(self, net_inputs, activations, label):
		changes_w, changes_b= [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
		error = self.cross_entropy_derivative(activations[-1], label)
		changes_b[-1], changes_w[-1] = error.T, np.dot(error.T, activations[-2].T)
		for i in range(2, len(self.layers_sizes)):
			sd = self.sigmoid_derivative(net_inputs[-i])
			error = np.dot(self.weights[-i + 1].T, error.T) * sd
			changes_b[-i], changes_w[-i] = error, np.dot(error, activations[-i - 1].T)
		return changes_b, changes_w

	@staticmethod
	def sigmoid_derivative(z):
		return sigmoid_activation(z) * (1 - sigmoid_activation(z))

	def train(self):
		self.convert_in_one_hot_vector(self.train_set_label)
		for i in range(self.no_data_samples):
			net_inputs, activations, output = self.feed_forward(self.train_set_input[i])
			changes_b, changes_w = self.backward(net_inputs, activations, self.train_set_label[i])
			self.weights = [w - nw * self.learning_rate for w, nw in zip(self.weights, changes_w)]
			self.biases = [b - nb * self.learning_rate for b, nb in zip(self.biases, changes_b)]
		self.accuracy()


	def accuracy(self):
		self.convert_in_one_hot_vector(self.test_set_label)
		results = [np.argmax(self.feed_forward(x)[2]) for x, y in zip(self.test_set_input, self.test_set_label)]
		print(sum(int(x == y) for x, y in zip(results, self.test_set_label))/ len((self.test_set)) * 100)
