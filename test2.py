from matplotlib import pyplot as plt

from model.builder import build_tree_horizontal
from train import holeaggregator

# 构建两层树结构
from train.optimizer import Optimizer

layers = build_tree_horizontal('data/full', [100, 200, 100], [[3]])
c1, c2, c3 = layers[0]
root = layers[1][0]
c2.verbose = 1

# master = layers[2][0]

# # 聚合算法1测试
# root.set_aggregator(agg1.Agg1())
# root.train(epoch=18, asy=False, lambda_2=0.12, lambda_1=0.12, init_lr=0.02, trans_delay=0.0, fake_foreign=False)
#
# # c1.history.plot('c1')
# # c2.history.plot('c2')
# # c3.history.plot('c3')
# root.history.plot('agg-sum')

agg = holeaggregator.HoleAggregator()

opt = Optimizer()

root.set_aggregator(agg)
root.optimizer = opt


# agg.interval_decay_rate = 0.0
# agg.initial_interval = 1
# root.reset()
# root.train(epoch=1000, asy=False, lambda_2=0.12, lambda_1=0.12, init_lr=0.015, trans_delay=0.05, fake_foreign=False)
# root.history.plot('E=1')

# root.reset()
# root.train(epoch=30, asy=False, lambda_2=0.16, lambda_1=0.16, init_lr=0.02, trans_delay=0.05, fake_foreign=False)
# root.history.plot('C=1.0')
#
# root.reset()
# root.grad_max = 0.6
# root.train(epoch=30, asy=False, lambda_2=0.16, lambda_1=0.16, init_lr=0.02, trans_delay=0.05, fake_foreign=False)
# root.history.plot('C=0.6')
# root.reset()
def set_epsilon(epsilon):
    c1.epsilon = epsilon
    c2.epsilon = epsilon
    c3.epsilon = epsilon


set_epsilon(6)
root.grad_max = 0.5
opt.lr_decay_rate = 0.07
agg.interval_decay_rate = 1 / 2.8
root.train(epoch=8, asy=False, lambda_2=0.06, lambda_1=0.06, init_lr=0.08, trans_delay=0.0, fake_foreign=False)

c2.history.plot('ε=6')

root.reset()
set_epsilon(4)
root.grad_max = 0.5
opt.lr_decay_rate = 0.04
agg.interval_decay_rate = 1 / 2.7
root.train(epoch=8, asy=False, lambda_2=0.06, lambda_1=0.06, init_lr=0.08, trans_delay=0.0, fake_foreign=False)
c2.history.plot('ε=4')

root.reset()
set_epsilon(3)
root.grad_max = 0.5
opt.lr_decay_rate = 0.1
agg.interval_decay_rate = 1 / 6.6
root.train(epoch=8, asy=False, lambda_2=0.06, lambda_1=0.06, init_lr=0.08, trans_delay=0.0, fake_foreign=False)
c2.history.plot('ε=3')

# root.reset()
# agg.interval_decay_rate = 1/3.3
# agg.initial_interval = 2
# root.train(epoch=10, asy=False, lambda_2=0.1201, lambda_1=0.1201, init_lr=0.02, trans_delay=0.05, fake_foreign=False)
# root.history.plot('E-increment')

plt.legend()
plt.savefig('unbalanced-E.jpg')
