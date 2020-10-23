import random

import numpy as np


class Percepron:
    def __init__(self, train_set, test_set):
        self.no_of_epochs = 10
        self.test_set = test_set
        self.test_set_input = test_set[0]
        self.test_set_label = test_set[1]
        self.train_set_input = train_set[0]
        self.train_set_label = train_set[1]
        self.len_training_set = len(self.train_set_input)
        self.len_features = len(self.train_set_input[0])
        self.mini_batch_len = 1024
        self.learning_rate = 0.0002

    def set_weights(self):
        self.weights = np.random.rand(self.len_features, 1)

    def set_bias(self):
        self.bias = np.random.rand(self.len_features, 1)

    def net_input(self, x, weights, bias):
        return sum(np.dot(x, weights) + bias)

    def activation(self, z):
        return 0 if z < 0 else 1

    def update(self, target, predict, x):
        weight_temp = np.zeros(np.shape(self.weights))
        weight_temp[:, 0] = self.learning_rate * (target - predict) * x
        bias_temp = np.zeros(np.shape(self.bias))
        bias_temp[:, 0] = self.learning_rate * (target - predict)
        self.weights = weight_temp + self.weights
        self.bias = bias_temp + self.bias

    def training_phase_perceptron(self, value):
        self.set_bias()
        self.set_weights()
        for _ in range(self.no_of_epochs):
            mini_batch = random.sample(range(0, self.len_training_set), self.mini_batch_len)
            for j in mini_batch:
                if self.train_set_label[j] == value:
                    target = 1
                else:
                    target = 0
                predict = self.activation(self.net_input(self.train_set_input[j], self.weights, self.bias))
                self.update(target, predict, self.train_set_input[j])

    def test_phase_perceptron(self, value):
        correct = 0
        self.training_phase_perceptron(value)
        print("Accuracy {}".format(value))
        for i in range(len(self.test_set_input)):
            if value == self.test_set_label[i]:
                target = 1
            else:
                target = 0
            if target == self.activation(self.net_input(self.test_set_input[i], self.weights, self.bias)):
                correct += 1
            print(self.test_set_label[i], self.activation(self.net_input(self.test_set_input[i], self.weights, self.bias)))
        print(correct / len(self.test_set_input))

    def train_for_all(self):
        weights = []
        biases = []
        for i in range(0, 10):
            self.training_phase_perceptron(i)
            weights.append(self.weights)
            biases.append(self.bias)
        return np.asarray(weights), np.asarray(biases)

    def test_individual_all(self):
        for i in range(1):
            self.test_phase_perceptron(i)

    def test_all(self):
        correct = 0
        weights, biases = self.train_for_all()
        for i in range(len(self.test_set_input)):
            predicts = []
            for j in range(0, 10):
                predict = self.activation(self.net_input(self.test_set_input[i], weights[j], biases[j]))
                if predict == 1:
                    predicts.append(j)
            chosen = max(predicts, key=lambda x: self.net_input(self.test_set_input[i], weights[x], biases[x]), default = -1)
            if chosen == self.test_set_label[i]:
                correct += 1
        accuracy = correct / ((len(self.test_set_input)))
        return accuracy
