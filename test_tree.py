# coding=utf-8
from matplotlib import pyplot as plt

import train.aggregator
from model.builder import build_tree_horizontal
from exp import agg6, agg4

# 构建两层树结构

tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[2, 2, 2, 2], [4]], name_prefix='t1_')
root_1 = tree1.root
root_1.aggregator.initial_interval = 4
root_1.verbose = 1
root_1.expand_to_children()
tree1.get_at(0, 0).verbose = 1
tree1.get_at(1, 0).verbose = 1

root_1.train(epoch=10, init_lr=0.09, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False)
root_1.history.plot('3-layer-root')
tree1.get_at(0, 1).history.plot('3-layer-c1')
# tree1.get_at(1, 0).history.plot('3-layer-r1')
# tree1.get_at(1, 1).history.plot('3-layer-r2')


tree3 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[2, 2, 2, 2], [2, 2], [2]],
                              name_prefix='t3_')
root_3 = tree3.root
root_3.verbose = 1
tree3.get_at(0, 0).verbose = 1
root_3.expand_to_children()
root_3.train(epoch=4, init_lr=0.09, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False)
tree3.get_at(0, 1).history.plot('4-layer-c1')

root_3.history.plot('4-layer-root')

tree2 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]], name_prefix='t2_')
root_2 = tree2.root
tree2.get_at(0, 0).verbose = 1
root_2.verbose = 1
root_2.expand_to_children()
root_2.train(epoch=19, init_lr=0.09, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False)
tree2.get_at(0, 1).history.plot('2-layer-c1')

root_2.history.plot('2-layer-root')
plt.legend()

plt.legend()

plt.savefig('layers-compare')
