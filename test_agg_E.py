# coding=utf-8
from matplotlib import pyplot as plt

import train.aggs
from model.builder import build_tree_horizontal

# FedSGD
from model.leaf import Leaf
from train.aggs.adaptiveaggregator import AdaptiveAggregator

tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]], name_prefix='t1')
tree1.set_aggregator(AdaptiveAggregator())
root_1 = tree1.root
root_1.verbose = 1
root_1.expand_to_children()
root_1.optimizer.lr_decay_rate = 0
root_1.train(epoch=25, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_1.history.plot('E=adaptive')


tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]], name_prefix='t1')
root_1 = tree1.root
root_1.verbose = 1
root_1.expand_to_children()
root_1.optimizer.lr_decay_rate = 0
root_1.aggregator.interval_decay_rate = 0.7
root_1.train(epoch=25, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_1.history.plot('E=classic')

plt.legend()
plt.savefig('E-agg-compare')
