from model.leaf import Leaf
from train.aggs.holeaggregator import HoleAggregator
import numpy as np


class AdaptiveAggregator(HoleAggregator):

    def __init__(self):
        HoleAggregator.__init__(self)
        self.init_loss = 0.0
        self.last_ste = 0

    def round_start(self):
        self.init_loss = self.node.RMS(self.node.train_data)
        self.interval += 5  # = self.initial_interval

    def get_interval(self, epoch, total_epoch, child):
        height = self.node.height
        self.activate_time += 1
        if epoch != self.last_ste:
            loss = self.node.RMS(self.node.train_data)
            eta = np.sqrt(loss / self.init_loss)
            if self.interval < 10:
                eta = eta * 1.5
            self.interval *= eta
            self.last_ste = epoch
        E = int(max(2, self.interval))
        if isinstance(child, Leaf):
            #print('{}->{},interval={}'.format(self.node.name, child.name, E))
            return E
        else:
            #print('{}->{},interval={}'.format(self.node.name, child.name, int(2 + np.tanh(E) / height)))
            return int(2 + (np.tanh(E) / height))
