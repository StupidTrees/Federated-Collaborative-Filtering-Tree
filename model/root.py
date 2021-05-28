import copy
import multiprocessing
import random
import time

import numpy as np

from model.node import Node
from train.aggregator import Aggregator


class Root(Node):
    """
    根节点，继承自Node
    """

    def __init__(self, name, k=10, clients=None, verbose=0, aggregator=None, optimizer=None, expand_children=False):
        Node.__init__(self, name, k=k, clients=clients, verbose=verbose, optimizer=optimizer)
        self.expand_children = expand_children
        if aggregator is None:
            aggregator = Aggregator()
        aggregator.node = self
        self.aggregator = aggregator
        self.reset()

    def predict(self, uid, iid):
        for child in self.children:
            pre = child.predict(uid, iid)
            if pre > 0:
                return pre
        return -1

    def update_v(self, v_map):
        for iid in self.item_map.keys():
            if iid in v_map.keys():
                self.item_map[iid] = v_map[iid]
        for child in self.children:
            child.update_v(v_map)

    def RMS(self, batch):
        result = 0.0
        count = len(batch)
        for (uid, iid, r) in batch:
            tmp_sum = 0.0
            tmp_num = 0
            for child in self.children:
                pr = child.predict(uid, iid)
                if pr > 0:
                    tmp_sum += np.square((float(r) / 5) - pr)
                    tmp_num += 1
            if tmp_num == 0:
                result += 0
            else:
                result += tmp_sum / tmp_num
        return np.sqrt(result / count)

    def reset(self):
        Node.reset(self)
        self.optimizer.reset()
        self.aggregator.reset()
        self.item_map.clear()
        self.test_data.clear()
        for child in self.children:
            child.reset()

        for child in self.children:
            child.parent = self
            if child.height + 1 > self.height:
                self.height = child.height + 1
            for iid in child.item_map.keys():
                if iid not in self.item_map.keys():
                    self.item_map[iid] = np.random.normal(0, 0.1, self.K)
            self.test_data += child.test_data
        for child in self.children:
            if self.expand_children:
                child.expand_v(self.item_map.keys())
        random.shuffle(self.test_data)
        self.test_data = self.test_data[0:int(len(self.test_data) / 2) - 1]  # self.test_data

    def do_train(self, epoch=10, parent_ste=0, init_lr=0.005, asy=True,
                 trans_delay=0.5, lambda_1=0.1, lambda_2=0.1, queue=None, fake_foreign=False):
        start_time = time.time()
        for child in self.children:
            child.update_v(self.item_map)
        v_old = copy.deepcopy(self.item_map)
        self.optimizer.round_begin(init_lr)
        for ste in range(epoch):
            q = multiprocessing.Queue(len(self.children))
            all_gradients = {}
            children_participate = self.aggregator.draw(epoch)
            for child in children_participate:
                gradients = child.train(asy=asy, init_lr=self.optimizer.get_child_init_lr(init_lr, epoch, child),
                                        epoch=self.aggregator.interval(epoch, child), queue=q,
                                        lambda_1=lambda_1,
                                        lambda_2=lambda_2)
                if not asy:
                    time.sleep(trans_delay)
                    all_gradients[child.name] = gradients
            if asy:
                for _ in range(len(self.children)):
                    client_name, gradients = q.get(block=True)
                    time.sleep(trans_delay)
                    all_gradients[client_name] = gradients
            self.aggregator.aggregate(epoch, children_participate, all_gradients)
            self.aggregator.dispatch(epoch, children_participate)
            rms = self.RMS(self.test_data)
            if self.verbose > 0:
                print('{}||epoch{},rms={}'.format(self.name, ste, rms))
            self.history.add(time.time() - start_time, rms)
        total_gradients_v = {iid: (self.item_map[iid] - v_old[iid]) for iid in self.item_map.keys()}
        if asy and queue:
            queue.put((self.name, total_gradients_v))
        return total_gradients_v
