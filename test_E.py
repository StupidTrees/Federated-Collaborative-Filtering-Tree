# coding=utf-8
from matplotlib import pyplot as plt

import train.aggs
from model.builder import build_tree_horizontal

# FedSGD
from model.leaf import Leaf

tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]],name_prefix='t1')
root_1 = tree1.root
root_1.verbose=1
root_1.aggregator.initial_interval = 1
root_1.aggregator.interval_decay_rate = 1
root_1.aggregator.soft = False
root_1.expand_to_children()
root_1.optimizer.lr_decay_rate = 0
root_1.train(epoch=150, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_1.history.plot('E=1')

# FedAVG
tree2 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]],name_prefix='t2')
root_2 = tree2.root
root_2.aggregator.initial_interval = 10
root_2.aggregator.interval_decay_rate = 1
root_2.aggregator.soft = False
root_2.verbose=1
root_2.expand_to_children()
root_2.optimizer.lr_decay_rate = 0
root_2.train(epoch=40, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_2.history.plot('E=10')

# Centralize
tree3 = build_tree_horizontal('data/full', [80], [[1]],name_prefix='t3')
root_3 = tree3.get_at(0,0)
root_3.verbose = 1
root_3.train_data = root_1.train_data
root_3.test_data = root_1.test_data
root_3.optimizer.lr_decay_rate = 0
root_3.train(epoch=100, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_3.history.plot('centralized')

plt.legend()
plt.savefig('E-compare')
