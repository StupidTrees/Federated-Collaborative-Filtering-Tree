import copy
import multiprocessing
import random
import time

import numpy as np

from train.holeaggregator import HoleAggregator
from model.node import Node


class Root(Node):
    """
    根节点，继承自Node
    """

    def __init__(self, name, k=10, clients=None, verbose=0, aggregator=None, optimizer=None, expand_children=False):
        Node.__init__(self, name, k=k, clients=clients, verbose=verbose, optimizer=optimizer)
        self.expand_children = expand_children
        self.weight_map = {}
        if aggregator is None:
            aggregator = HoleAggregator()
        self.aggregator = aggregator
        self.set_aggregator(aggregator)
        self.bind_child()

    def get_weight(self, child_name):
        return self.weight_map[child_name] / sum(self.weight_map.values())

    def set_aggregator(self, aggregator):
        aggregator.node = self
        self.aggregator = aggregator

    def predict(self, uid, iid):
        for child in self.children:
            pre = child.predict(uid, iid)
            if pre > 0:
                return pre
        return -1

    def update_v(self, v_map, hole=True):
        for iid in self.item_map.keys():
            if iid in v_map.keys():
                self.item_map[iid] = np.copy(v_map[iid])
        self.aggregator.dispatch(-1, self.children, False)
        # for child in self.children:
        #     child.update_v(v_map, False)

    def RMS(self, batch):
        rmses = []
        weights = []
        for child in self.children:
            rmses.append(child.RMS(batch))
            weights.append(self.weight_map[child.name])
        # print('rms_agv,{}:{},{}'.format(self.name,rmses,weights))
        return np.average(rmses, weights=weights)

    def user_size(self):
        sum = 0
        for child in self.children:
            sum += child.user_size()
        return sum

    def bind_child(self):
        self.optimizer.reset()
        self.aggregator.reset()
        self.item_map.clear()
        self.test_data.clear()

        min_test_size = 999999
        for child in self.children:
            if len(child.test_data) < min_test_size:
                min_test_size = len(child.test_data)
            child.reset()
            child.parent = self
            self.weight_map[child.name] = child.user_size()

        for child in self.children:
            if child.height + 1 > self.height:
                self.height = child.height + 1
            for iid in child.item_map.keys():
                if iid not in self.item_map.keys():
                    self.item_map[iid] = np.random.normal(0, 0.1, self.K)
            self.test_data += child.test_data[0:min_test_size]  # 保证每个孩子贡献相同数量的test_data
        if self.expand_children:
            self.expand_to_children()
        random.shuffle(self.test_data)
        self.test_data = self.test_data[0:int(len(self.test_data) / 2) - 1]  # self.test_data

    def expand_to_children(self, assign_test_data=True, recursive=True):
        for child in self.children:
            child.expand_v(self.item_map.keys())
            if assign_test_data:
                child.test_data = self.test_data
            if recursive and isinstance(child, Root):
                child.expand_to_children(assign_test_data, recursive)

    def reset(self):
        Node.reset(self)
        self.optimizer.reset()
        self.aggregator.reset()
        for child in self.children:
            child.parent = self
            for iid in child.item_map.keys():
                if iid not in self.item_map.keys():
                    self.item_map[iid] = np.random.normal(0, 0.1, self.K)
            child.reset()

    def do_train(self, epoch=10, parent_ste=0, init_lr=0.005,
                 trans_delay=0.5, lambda_1=0.1, lambda_2=0.1, queue=None, fake_foreign=False):
        for child in self.children:
            child.update_v(self.item_map)
        v_old = copy.deepcopy(self.item_map)
        self.optimizer.round_begin(init_lr)
        for ste in range(epoch):

            q = multiprocessing.Queue(len(self.children))
            all_gradients = {}
            children_participate = self.aggregator.draw(epoch)
            for child in children_participate:
                gradients = child.train(init_lr=self.optimizer.get_child_init_lr(init_lr, epoch, child),
                                        epoch=self.aggregator.get_interval(ste, epoch, child), queue=q,
                                        lambda_1=lambda_1,
                                        lambda_2=lambda_2)
                if not self.aggregator.asy:
                    time.sleep(trans_delay)
                    all_gradients[child.name] = gradients
            if self.aggregator.asy:
                for _ in range(len(self.children)):
                    client_name, gradients = q.get(block=True)
                    time.sleep(trans_delay)
                    all_gradients[client_name] = gradients
            rms = self.RMS(self.test_data)
            if self.verbose > 0:
                print('{}||epoch{},rms={}'.format(self.name, ste, rms))
            self.history.add(time.time() - self.start_time, rms)
            if ste != epoch - 1:
                self.aggregator.aggregate(epoch, children_participate, 0.0, all_gradients)
                self.aggregator.dispatch(epoch, children_participate)

        # rms = self.RMS(self.test_data)
        # if self.verbose > 0:
        #     print('{}||epoch_final,rms={}'.format(self.name, rms))
        # self.history.add(time.time() - self.start_time, rms)

        def cut_grad(dv):
            return dv
            # dv_cut = np.minimum(np.ones(self.K) * self.grad_max, dv)
            # return np.maximum(-np.ones(self.K) * self.grad_max, dv_cut)

        total_gradients_v = {iid: cut_grad(self.item_map[iid] - v_old[iid]) for iid in self.item_map.keys()}

        if self.parent and self.parent.aggregator.asy and queue:
            queue.put((self.name, total_gradients_v))
        return total_gradients_v
