import pickle, gzip
import matplotlib.pyplot as plt

from SimpleNeuralNetwork import SimpleNeuralNetwork

def read_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()
    return train_set, valid_set, test_set


if __name__ == '__main__':
    train_set, valid_set, test_set = read_dataset()
    features_set, label_set = train_set[0], train_set[1]
    neural = SimpleNeuralNetwork(features_set, label_set)
    print(neural.online_training())