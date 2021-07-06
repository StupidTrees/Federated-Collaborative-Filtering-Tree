import numpy as np

from train.aggregator import Aggregator


class Agg6(Aggregator):
    # def aggregate(self, epoch, child_participate, all_gradients):
    #     total_gradients_v = {}
    #     for child_name, gradients in all_gradients.items():
    #         for (iid, grad) in gradients.items():
    #             # print('max:{},min:{}'.format(np.max(grad),np.min(grad)))
    #             if iid not in total_gradients_v.keys():
    #                 total_gradients_v[iid] = []
    #             total_gradients_v[iid].append((child_name, grad))
    #     for (iid, pairs) in total_gradients_v.items():
    #
    #
    #         self.node.item_map[iid] += np.average(gradients, axis=0, weights=weights)
    #     return
    def get_interval(self, epoch, total_epoch, child):
        """
        决定让某个孩子独自训练多少轮（E）
        :param epoch: 当前节点的训练轮次
        :param total_epoch 当前节点需要的总训练伦茨
        :param child: 某个孩子
        :return: 孩子在本轮训练中，自己要进行多少轮训练
        """
        self.activate_time += 1
        self.interval = self.initial_interval*(0.7**epoch)  # + int(self.interval_increase_rate * self.activate_time)
        return int(self.interval)

    def dispatch(self, epoch, children_participate):
        for child in children_participate:
            v_map_specified = {}
            for iid, grad in self.total_grad_map.items():
                # real_grad = grad
                # if iid in self.child_grad_map[child.name].keys():
                #     real_grad -=
                make_it_up = 0
                if iid in self.child_grad_map[child.name].keys():
                    make_it_up = self.child_grad_map[child.name][iid]
                v_map_specified[iid] = self.node.item_map[iid]# + (1.8 * self.node.get_weight(child.name)) * make_it_up
            child.update_v(v_map_specified)
