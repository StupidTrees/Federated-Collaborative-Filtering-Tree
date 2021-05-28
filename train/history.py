import os

from matplotlib import pyplot as plt


class History:
    """
    用来记录训练历史的类
    """

    def __init__(self, x=None, y=None):
        if x is None:
            x = []
        if y is None:
            y = []
        self.Y = y
        self.X = x

    def add(self, x, y):
        self.X.append(x)
        self.Y.append(y)

    def reset(self):
        self.X.clear()
        self.Y.clear()

    def save(self, file):
        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file, 'w+') as f:
            for (xi, yi) in zip(self.X, self.Y):
                f.write('{}\t{}\n'.format(xi, yi))
            f.close()

    def plot(self, name):
        plt.plot(self.X, self.Y, label=name)


def read_history(file):
    x = []
    y = []
    with open(file) as f:
        for line in f.readlines():
            xi, yi = line.split()
            x.append(float(xi))
            y.append(float(yi))
    return History(x, y)
