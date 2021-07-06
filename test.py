import random

from model.leaf import Leaf
from model.node import *
from matplotlib import pyplot as plt

x0 = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # , 256]
names = [str(i) + '' for i in x0]

y = [21.375601, 21.188999, 21.020800, 21.353600, 21.429800, 25.801102, 28.200001, 39.544098, 35.245201]
x = range(len(y))
growth = [max(y) - 3 + y[i] - y[i - 1] for i in range(1, len(y))]
plt.fill_between(x, y, [min(y) - 1 for x in y], label='duration')
plt.plot(x[1:], growth, label='growth', color='orange')
plt.xticks(x, names, rotation=45)
plt.legend()
plt.show()
