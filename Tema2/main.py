import pickle, gzip
from Percepron import Percepron


def read_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()
    return train_set, valid_set, test_set


if __name__ == '__main__':
    train_set, valid_set, test_set = read_dataset()
    neural = Percepron(train_set, test_set[0])
    print(neural.test_individual_all())