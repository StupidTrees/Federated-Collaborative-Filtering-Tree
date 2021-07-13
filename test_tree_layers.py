# coding=utf-8
from matplotlib import pyplot as plt

import train.aggregator
from model.builder import build_tree_horizontal
from exp import agg6, agg4

# 比较层数对训练效果的影响

tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[4, 4], [2]],
                              name_prefix='t1_')
root_1 = tree1.root
root_1.verbose = 1
root_1.expand_to_children()
root_1.train(epoch=10, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
root_1.history.plot('3-layer-root')


tree2 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]], name_prefix='t2_')
root_2 = tree2.root
root_2.verbose = 1
root_2.expand_to_children()
root_2.train(epoch=10, init_lr=0.01, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True)
tree2.get_at(0, 1).history.plot('2-layer-c1')

root_2.history.plot('2-layer-root')
plt.legend()

plt.savefig('layers-compare')
