import random

import numpy as np

class Percepron:
    def __init__(self, train_set, test_set, label_encode = False, just_result = False):
        self.just_result = just_result
        self.label_encode = label_encode
        self.no_of_epochs = 50
        self.test_set = test_set
        self.test_set_input = np.hstack((np.ones((len(test_set[0]) , 1)), test_set[0]))
        self.test_set_label = test_set[1]
        self.train_set_input = np.hstack((np.ones((len(train_set[0]) , 1)), train_set[0]))
        self.train_set_label = train_set[1]
        self.len_training_set = len(self.train_set_input)
        self.len_features = len(self.train_set_input[0])
        self.mini_batch_len = 1024
        self.learning_rate = 0.3

    def get_one_hot_vector(self, labels):
        res = np.eye(10)[np.array(labels).reshape(-1)]
        return res.reshape(list(labels.shape) + [10])

    def convert_in_one_hot_vector(self):
        self.test_set_label = self.get_one_hot_vector(self.test_set_label)
        print(self.test_set_label)

    def set_weights(self):
        self.weights = np.random.rand(self.len_features, 1)

    def net_input(self, x, weights):
        return sum(np.dot(x, weights))

    def activation(self, z):
        return 0 if z < 0 else 1

    def training_phase_perceptron(self, value):
        self.set_weights()
        for _ in range(self.no_of_epochs):
            mini_batch = random.sample(range(0, self.len_training_set), self.mini_batch_len)
            weight_temp = np.zeros(np.shape(self.weights))
            for j in mini_batch:
                target = self.get_target(value, self.train_set_label[j])
                predict = self.activation(self.net_input(self.train_set_input[j], self.weights))
                weight_temp[:, 0] += self.learning_rate * (target - predict) * self.train_set_input[j]
            self.weights = weight_temp + self.weights

    def get_target(self, value, label):
        return 1 if label == value else 0

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
        print(correct / len(self.test_set_input) * 100)

    def train_for_all(self):
        weights = []
        for i in range(10):
            self.training_phase_perceptron(i)
            weights.append(self.weights[:, 0])
        return np.asarray(weights)

    def test_individual_all(self):
        for i in range(10):
            self.test_phase_perceptron(i)

    def test_all(self):
        weights = self.train_for_all()
        results = np.dot(self.test_set_input, np.transpose(weights))
        predicts = np.argmax(results, axis=1)
        one_hot_vectors_predicts = np.zeros((predicts.size, 10),dtype=np.int32)
        one_hot_vectors_predicts[np.arange(predicts.size), predicts] = 1
        print(one_hot_vectors_predicts)
        if not self.just_result:
            if self.label_encode == False:
                self.convert_in_one_hot_vector()
            print(one_hot_vectors_predicts)
            correct = np.sum(np.all(one_hot_vectors_predicts == self.test_set_label, axis=1))
            accuracy = correct / (len(self.test_set_label)) * 100
            return accuracy
