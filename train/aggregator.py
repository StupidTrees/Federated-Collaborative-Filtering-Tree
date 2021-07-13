import random

import numpy as np

from model.leaf import Leaf


class Aggregator:
    """
    Aggregator对象，用于控制任意一个根节点的联邦策略
    可以在Root对象初始化时为其指定Aggregator

    """

    def __init__(self, asy=False, interval=60, interval_decay_rate=1.1):
        self.initial_interval = interval
        self.asy = asy
        self.interval_decay_rate = interval_decay_rate
        self.total_grad_list_map = {}  # {iid：[(cname1:grad1), (cname2:grad2)]}
        self.total_grad_map = {}  # {iid:grad, iid2:grad}
        self.child_grad_map = {}  # {cname1: grad, cname2: grad}
        self.interval = float(interval)
        self.activate_time = 0
        self.node = None

    def reset(self):
        self.total_grad_list_map = {}
        self.child_grad_map = {}
        self.activate_time = 0
        self.interval = self.initial_interval

    def draw(self, epoch):
        """
        从所有子节点里选出要参与的节点
        :param epoch: 当前训练的轮次
        :return: 选择后的子节点列表，默认为全部
        """
        percentage = 1.0
        from model.root import Root
        if isinstance(self.node, Root):
            return random.sample(self.node.children, int(len(self.node.children) * percentage))
        return self.node.children

    def get_interval(self, epoch, total_epoch, child):
        """
        决定让某个孩子独自训练多少轮（E）
        :param epoch: 当前节点的训练轮次
        :param total_epoch 当前节点需要的总训练伦茨
        :param child: 某个孩子
        :return: 孩子在本轮训练中，自己要进行多少轮训练
        """
        height = self.node.height
        self.activate_time += 1
        self.interval = self.initial_interval * (self.interval_decay_rate ** epoch)
        rs = int(self.interval)#max(int(np.sqrt(total_epoch)), int(self.interval))
        if isinstance(child, Leaf):
            print('{}->{},interval={}'.format(self.node.name,child.name,int(rs*len(self.node.children)/8)))
            return int(rs*len(self.node.children)/8)
        else:
            print('{}->{},interval={}'.format(self.node.name,child.name,int(1 + np.tanh(rs) / height)))
            return int(1 + (np.log2(rs) / height))

    def aggregate(self, epoch, child_participate, loss, all_gradients):
        """
        梯度（权值)聚合操作
        :param epoch: 当前训练的轮次
        :param child_participate: 参与本次训练的孩子列表
        :param all_gradients: 所有参与的子节点返回的梯度结果
            格式为{孩子名称1：{商品1：[梯度1，梯度2...]，商品2:[梯度1，梯度2]}，孩子名称2:...}
        :return:
        """
        for cname, gradients in all_gradients.items():
            if cname not in self.child_grad_map.keys():
                self.child_grad_map[cname] = {}
            for (iid, grad) in gradients.items():
                if iid not in self.total_grad_list_map.keys():
                    self.total_grad_list_map[iid] = []
                    self.total_grad_map[iid] = 0.0
                self.total_grad_list_map[iid].append((cname, grad))
                self.child_grad_map[cname][iid] = grad
        self.do_aggregate(loss)

    def do_aggregate(self, loss):
        for (iid, pairs) in self.total_grad_list_map.items():
            weights = [self.node.weight_map[cn] for cn, g in pairs]
            gradients = [g for cn, g in pairs]
            self.total_grad_map[iid] += np.sum(gradients, axis=0)#, weights=weights)
            self.node.item_map[iid] += np.sum(gradients, axis=0)#, weights=weights)

    def dispatch(self, epoch, children_participate, from_upper=False):
        """
        权值分发操作
        :param epoch: 当前训练的轮次
        :param children_participate: 参与本次训练的孩子列表
        :param from_upper: 是否是来自更上一层父节点的递归调用
        :return:
        """
        for child in children_participate:
            v_map_specified = {}
            # for iid, grad in self.total_grad_map.items():
            #     v_map_specified[iid] = self.node.item_map[iid]
            child.update_v(self.node.item_map)
