# coding=utf-8
from matplotlib import pyplot as plt

from model.builder import build_tree_horizontal


# 用于测试树端宽度对收敛效果的影响

def create_tree(total=80, width=4, expand_to_children=True):
    clist = []
    for i in range(width):
        clist.append(total / width)
    tree = build_tree_horizontal('data/full', clist, [[width]], name_prefix='t2_')
    if expand_to_children:
        tree.root.expand_to_children()
    return tree


tree2 = create_tree(width=2)
tree2.root.train(epoch=10, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
                 fake_foreign=False)
tree2.get_at(0, 1).history.plot('2-c1')
tree2.root.history.plot('2-root')


tree2 = create_tree(width=2)
tree2.root.train(epoch=10, init_lr=0.008, lambda_1=0.06, lambda_2=0.06, trans_delay=0.0,
                 fake_foreign=False,adam=True)
tree2.get_at(0, 1).history.plot('2-c1-adam')
tree2.root.history.plot('2-root-adam')

plt.legend()
plt.savefig('adam-compare')
