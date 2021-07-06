from matplotlib import pyplot as plt

import train.aggregator
from model.builder import build_tree_horizontal
from exp import agg6, agg4

# 构建两层树结构

tree1 = build_tree_horizontal('data/full', [50, 50, 50, 50], [[2, 2], [2]], name_prefix='t1_')
root_1 = tree1.root
root_1.aggregator.initial_interval = 4
root_1.verbose = 1
root_1.expand_to_children()
tree1.get_at(1, 0).verbose = 1
tree1.get_at(1, 1).verbose = 1

root_1.train(epoch=9, lambda_2=0.09, lambda_1=0.09, init_lr=0.08, trans_delay=0.0, fake_foreign=False)
root_1.history.plot('3-layer')
tree1.get_at(1, 0).history.plot('3-layer-r1')
tree1.get_at(1, 1).history.plot('3-layer-r2')

tree2 = build_tree_horizontal('data/full', [50, 50, 50, 50], [[4]], name_prefix='t2_')
root_2 = tree2.root
root_2.verbose = 1
# tree2.get_at(0, 0).verbose=1
# tree2.get_at(0,1).verbose=1
# tree2.get_at(0, 2).verbose=1
# tree2.get_at(0, 3).verbose=1
root_2.expand_to_children()
root_2.train(epoch=16, lambda_2=0.09, lambda_1=0.09, init_lr=0.08, trans_delay=0.0, fake_foreign=False)
root_2.history.plot('2-layer')
# tree2.get_at(0, 0).history.plot('2-layer-c1')
# tree2.get_at(0,1).history.plot('2-layer-c2')
# tree2.get_at(0, 2).history.plot('2-layer-c3')
# tree2.get_at(0, 3).history.plot('2-layer-c4')
plt.legend()
plt.savefig('layers-compare')
