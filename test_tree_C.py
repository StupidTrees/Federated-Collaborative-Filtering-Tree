# coding=utf-8
from matplotlib import pyplot as plt

from model.builder import build_tree_horizontal

# 用于测试树端宽度对收敛效果的影响
tree1 = build_tree_horizontal('data/full', [10, 10, 10, 10, 10, 10, 10, 10], [[8]],
                              name_prefix='t1_')
tree1.set_grad_max(0.4)
root_1 = tree1.root
root_1.verbose = 1
root_1.expand_to_children()
root_1.aggregator.interval_decay_rate = 0.85
root_1.train(epoch=20, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True, adaptive_c=False)
root_1.history.plot('c=0.4')

root_1.reset()
tree1.set_grad_max(0.6)
root_1.train(epoch=20, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
             fake_foreign=False, adam=True, adaptive_c=True)
root_1.history.plot('c-adaptive-from 0.6')

plt.legend()
plt.savefig('C-compare')
