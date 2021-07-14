from threading import Thread

import numpy as np
import time
from train.history import History
from train.optimizer import Optimizer
from functools import cmp_to_key as ctk


class Node:
    """
    树的节点
    """

    def __init__(self, name, k=10, clients=None, verbose=0, optimizer=None, grad_max=0.6):

        if clients is None:
            clients = []
        self.name = name  # 名称
        self.K = k  # 隐特征维度
        self.history = History()  # 训练历史
        self.verbose = verbose  # 控制过程输出的标志
        self.children = clients  # 孩子列表
        self.test_data = []  # 测试数据集
        self.user_map = {}
        self.item_map = {}
        self.parent = None
        self.foreign = set()  # 外来特征id集合
        self.height = 0
        self.first_start = True
        self.start_time = time.time()
        self.grad_max = grad_max  # 梯度的最大值，裁剪梯度
        if optimizer is None:
            optimizer = Optimizer()
        self.optimizer = optimizer
        self.user_map_reversed = {y: x for (x, y) in self.user_map.items()}

    def user_size(self):
        pass

    def expand_v(self, iid_list):
        """
        扩展某个节点的item列表，即加入外来特征
        :param iid_list: 外来商品id列表
        """
        for iid in iid_list:
            if int(iid) not in self.item_map.keys():
                self.item_map[int(iid)] = np.random.normal(0, 0.1, (self.K,))
                self.foreign.add(iid)

    def RMS(self, batch):
        pass

    def reset(self):
        """
        重置状态（权重复位）
        """
        self.start_time = time.time()
        self.first_start = True
        for uid in self.user_map.keys():
            self.user_map[uid] = np.random.normal(0, 0.1, self.K)
        for iid in self.item_map.keys():
            self.item_map[iid] = np.random.normal(0, 0.1, self.K)
        self.history.reset()

    def train(self, queue=None,
              epoch=50,
              parent_ste=0,
              parent_epoch=0,
              init_lr=0.005,
              lambda_1=0.1,
              lambda_2=0.1,
              trans_delay=0.5, fake_foreign=False, adam=False, adaptive_c=True):
        """
        开始训练（同步或异步）
        :param queue: 异步训练时用于通信的阻塞队列
        :param epoch: 本节点训练总轮次
        :param parent_ste:当前父节点的轮次
        :param parent_epoch:当前父节点总训练轮次
        :param init_lr: 学习率初始值
        :param lambda_1: 迭代正则参数1
        :param lambda_2: 迭代正则参数2
        :param trans_delay: 模拟通信延时
        :param fake_foreign: 针对训练集覆盖不到的那些（用户，商品）对，是否引入假训练数据
        :return:
        """
        if self.parent and self.parent.aggregator.asy:
            thread = Thread(target=self.do_train,
                            args=(
                                epoch, parent_ste, parent_epoch, init_lr, trans_delay, lambda_1,
                                lambda_2,
                                queue, fake_foreign, adam, adaptive_c))
            thread.start()
        else:
            return self.do_train(epoch=epoch, parent_ste=parent_ste, parent_epoch=parent_epoch, init_lr=init_lr,
                                 trans_delay=trans_delay,
                                 lambda_1=lambda_1,
                                 lambda_2=lambda_2, queue=None, fake_foreign=fake_foreign, adam=adam,
                                 adaptive_c=adaptive_c)

    def do_train(self, epoch=10, parent_ste=0, parent_epoch=0, init_lr=0.005,
                 trans_delay=0.5, lambda_1=0.1, lambda_2=0.1, queue=None, fake_foreign=False, adam=False,
                 adaptive_c=True):
        if self.first_start:
            self.start_time = time.time()
            self.first_start = False

    def predict(self, uid, iid):
        pass

    def update_v(self, v_map):
        pass


def cmp_top_k(a, b):
    if a[1] == b[1]:
        return a[0] - b[0]
    else:
        return a[1] - b[1]
