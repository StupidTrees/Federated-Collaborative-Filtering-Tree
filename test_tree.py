from model.leaf import Leaf
from model.root import Root
from matplotlib import pyplot as plt

from train.aggregator import Aggregator

l1 = Leaf('c1', file_name='data/horizontal/full_U1', verbose=1)
l2 = Leaf('c2', file_name='data/horizontal/full_U2', verbose=0)
l3 = Leaf('c3', file_name='data/horizontal/full_U3', verbose=0)
l4 = Leaf('c4', file_name='data/horizontal/full_U4', verbose=0)
l5 = Leaf('c5', file_name='data/horizontal/full_U5', verbose=0)
l6 = Leaf('c6', file_name='data/horizontal/full_U6', verbose=0)

m1 = Root('m1', k=10, clients=[l1, l2], verbose=1, expand_children=True)
m2 = Root('m2', k=10, clients=[l3, l4], verbose=1, expand_children=True)
m3 = Root('m3', k=10, clients=[l5, l6], verbose=1, expand_children=True)

r = Root('root', k=10, clients=[m1, m2, m3], verbose=1)
r.train(epoch=8, asy=False, lambda_2=0.12, lambda_1=0.12, init_lr=0.03, trans_delay=0.0, fake_foreign=False)

# r.history.plot('root')
# m1.history.plot('m1')
# m2.history.plot('m2')
# m3.history.plot('m3')
l1.history.plot('l1')
# l4.history.plot('l4')
# l6.history.plot('l6')

l1.reset()
l1.test_data = m1.test_data
l1.train(epoch=200, asy=False, lambda_2=0.12, lambda_1=0.12, init_lr=0.02, trans_delay=0.01,
         fake_foreign=False)
l1.history.plot('l1-single')

plt.legend()  # 标签
plt.savefig('all.jpg')
