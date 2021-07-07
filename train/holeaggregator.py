import numpy as np

from model import leaf
from train.aggregator import Aggregator


class HoleAggregator(Aggregator):
    def dispatch(self, epoch, children_participate):
        for child in children_participate:
            v_map_specified = {}
            for iid, grad in self.total_grad_map.items():
                make_it_up = 0
                if iid in self.child_grad_map[child.name].keys() and isinstance(child, leaf.Leaf):
                    make_it_up = self.child_grad_map[child.name][iid]
                v_map_specified[iid] = self.node.item_map[iid] - self.node.get_weight(child.name) * make_it_up
            child.update_v(v_map_specified)
