import random

import numpy as np


class Aggregator:
    """
    Aggregator对象，用于控制任意一个根节点的联邦策略
    可以在Root对象初始化时为其指定Aggregator

    """

    def __init__(self, interval=1, interval_decay_rate=1 / 3.3):
        self.initial_interval = interval
        self.interval_decay_rate = interval_decay_rate
        self.node = None

    def reset(self):
        pass

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

    def interval(self, epoch, child):
        """
        决定让某个孩子独自训练多少轮（E）
        :param epoch: 当前节点的训练轮次
        :param child: 某个孩子
        :return: 孩子在本轮训练中，自己要进行多少轮训练
        """
        return self.initial_interval + int(self.interval_decay_rate * epoch)

    def aggregate(self, epoch, child_participate, all_gradients):
        """
        梯度（权值)聚合操作
        :param epoch: 当前训练的轮次
        :param child_participate: 参与本次训练的孩子列表
        :param all_gradients: 所有参与的子节点返回的梯度结果
            格式为{孩子名称1：{商品1：[梯度1，梯度2...]，商品2:[梯度1，梯度2]}，孩子名称2:...}
        :return:
        """
        total_gradients_v = {}
        for _, gradients in all_gradients.items():
            for (iid, grad) in gradients.items():
                if iid not in total_gradients_v.keys():
                    total_gradients_v[iid] = []
                total_gradients_v[iid].append(grad)
        for (iid, gradients) in total_gradients_v.items():
            self.node.item_map[iid] += np.average(gradients)
        return

    def dispatch(self, epoch, children_participate):
        """
        权值分发操作
        :param epoch: 当前训练的轮次
        :param children_participate: 参与本次训练的孩子列表
        :return:
        """
        for child in children_participate:
            child.update_v(self.node.item_map)
