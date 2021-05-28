from model.leaf import Leaf
from model.node import *
from matplotlib import pyplot as plt

# server.reset()
# times2, rmses2 = server.train(epoch=50, interval=1, asy=True)
# plt.plot(times2, rmses2, label='n=1')
# plt.show()

# children = []
# for i in range(943):
#     children.append(Leaf('c{}'.format(i + 1), 'data/one/full_U{}'.format(i + 1)))
# server = Leaf('server', 'data/full', K=10, clients=children, leaf=False, verbose=1)
# times, rmses = server.train(epoch=15, interval=1, asy=True, trans_delay=0.001)
# plt.plot(times, rmses, label='n=5')

#

# children = [Leaf('c1', 'data/full_U1_I1'),
#             Leaf('c2', 'data/full_U2_I2'),
#             Leaf('c3', 'data/full_U1_I2'),
#             Leaf('c4', 'data/full_U2_I1')]
# server = Root('server', 'data/full', k=10, clients=children, verbose=1)
#
# server.reset()
# his = server.train(epoch=400, interval=1, asy=True, lambda_2=0.12, lambda_1=0.12, eta=0.008, trans_delay=0.0,
#                    interval_factor=0.001)
# his.plot('fac=0,eta==0.08')
# his.save('data/output/a1')

# server.reset()
# his = server.train(epoch=60, interval=1, asy=True, lambda_2=0.12, lambda_1=0.12, eta=0.08, trans_delay=0.001,
#                   interval_factor=1 / 3, eta_decay_rate=0.05)
# his.plot('fac=1/3,decay=0.01')
# his.save('data/output/a2')
from model.root import Root

l1 = Leaf('c1', 'data/horizontal/full_U1', verbose=1)
l2 = Leaf('c2', 'data/horizontal/full_U2')
l3 = Leaf('c3', 'data/horizontal/full_U3')
l4 = Leaf('c4', 'data/horizontal/full_U4')
l5 = Leaf('c5', 'data/horizontal/full_U5')
l6 = Leaf('c6', 'data/horizontal/full_U6')
l7 = Leaf('c7', 'data/horizontal/full_U7')

server = Root('server', k=10, clients=[l1, l2, l3, l4, l5, l6, l7], verbose=1)

# 启用全特征
# for child in server.children:
#     child.expand_v(server.item_map.keys())

server.train(epoch=50, interval=1, asy=False, lambda_2=0.12, lambda_1=0.12, eta=0.02, trans_delay=0.000,
             interval_factor=1 / 3.3, eta_decay_rate=0.08, fake_foreign=False)
server.history.plot('total')

server.reset()
server.train(epoch=50, interval=1, asy=True, lambda_2=0.12, lambda_1=0.12, eta=0.02, trans_delay=0.000,
             interval_factor=1 / 3.3, eta_decay_rate=0.08, fake_foreign=True)
server.history.plot('total-fake-foreign')
# l1.history.plot('l1-federated')
#
# l1.reset()
# l1.train(epoch=1200, interval=1, asy=False, lambda_2=0.12, lambda_1=0.12, eta=0.02, trans_delay=0.000,
#          eta_decay_rate=0.0008)
# l1.history.plot('l1-single')

# server2 = Leaf('compact', 'data/full', k=10, verbose=1)
# server2.train(epoch=400, interval=1, asy=False, eta=0.08, trans_delay=0)
# his = server2.history
# his.plot('compact')
# his.save('data/output/a3')

plt.legend()  # 标签
plt.savefig('all.jpg')
