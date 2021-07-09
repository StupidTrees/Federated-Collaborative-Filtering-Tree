from model.node import *
from train.adam import Adam
from train.data import load_data
import tensorflow as tf


class Leaf(Node):
    """
    叶子节点，继承自Node
    """

    def __init__(self, name, data_tuple=None, file_name=None, k=10, verbose=0, optimizer=None, epsilon=999):
        Node.__init__(self, name, k=k, clients=None, verbose=verbose, optimizer=optimizer)

        if data_tuple is None:
            self.train_data, self.test_data, users, items = load_data(file=file_name,
                                                                      shuffle=True,
                                                                      batch_size=0,
                                                                      tensor=False,
                                                                      test_ratio=0.3)
        else:
            data, users, items = data_tuple
            self.train_data = data[0:int(len(data) * 0.8)]
            self.test_data = data[int(len(data) * 0.8):]
        for uid in users:
            self.user_map[uid] = np.random.normal(0, 0.1, k)
        for iid in items:
            self.item_map[iid] = np.random.normal(0, 0.1, k)

        self.average_rating = {}
        self.epsilon = epsilon
        self.hole_aggregate = True  # 是否为挖洞aggregate策略
        self.tmp_gradient_map = {}
        sum_map = {}
        for uid in users:
            sum_map[uid] = []
        for (uid, iid, r) in self.train_data:
            sum_map[uid].append(r / 5.0)
        for uid, ratings in sum_map.items():
            if len(ratings) > 0:
                self.average_rating[uid] = np.average(ratings)
            else:
                self.average_rating[uid] = 0.0

    def predict(self, uid, iid):
        if uid in self.user_map.keys() and iid in self.item_map.keys():
            return np.dot(self.user_map[uid], self.item_map[iid].T)
        return -1

    def reset(self):
        Node.reset(self)
        self.optimizer.reset()

    def expand_v(self, iid_list):
        Node.expand_v(self, iid_list)
        for iid in iid_list:
            if iid in self.foreign:
                for uid in self.user_map.keys():
                    self.train_data.append((uid, iid, self.average_rating[uid]))

    def RMS(self, batch):
        sum = 0.0
        count = 0
        for (uid, iid, r) in batch:
            if uid in self.user_map.keys() and iid in self.item_map.keys():
                pr = self.predict(uid, iid)
                count += 1
                sum += np.square((float(r) / 5) - pr)
        if count == 0:
            return 0
        return np.sqrt(sum / count)

    def user_size(self):
        return len(self.user_map.keys())

    def update_v(self, v_map, hole=True):
        for iid in self.item_map.keys():
            if iid in v_map.keys():
                self.item_map[iid] = np.copy(v_map[iid])
                if self.hole_aggregate and iid in self.tmp_gradient_map.keys() and hole:  # 挖洞策略，说明更新的v不包括自己上传的梯度，因此要再做一次自己的v更新
                    self.item_map[iid] += self.tmp_gradient_map[iid]
        if self.verbose > 0:
            print('{}||update,upper:{},rms={}'.format(self.name, not hole, self.RMS(self.test_data)))

    def apply_dp(self, iid, v_vec):
        """
        对v向量进行加噪处理，加入拉普拉斯分布的随机噪声使其满足epsilon-差分隐私
        :param v_vec: v向量
        :return: 差分后的
        """
        self.tmp_gradient_map[iid] = v_vec
        dv_cut = np.minimum(np.ones(self.K) * self.grad_max, v_vec)
        dv_cut = np.maximum(-np.ones(self.K) * self.grad_max, dv_cut)
        return dv_cut + np.random.laplace(0, 2 * self.grad_max / self.epsilon, self.K)

    def do_train(self, epoch=10, parent_ste=0, parent_epoch=0, init_lr=0.005,
                 trans_delay=0.5, lambda_1=0.1, lambda_2=0.1, queue=None, fake_foreign=False):
        Node.do_train(self)
        v_old = {iid: np.copy(vec) for iid, vec in self.item_map.items()}
        self.optimizer.round_begin(init_lr=init_lr)
        # 为每个向量设置一个Adam优化器
        v_adams_map = {iid: Adam(weights=v, lr=init_lr) for iid, v in self.item_map.items()}
        u_adams_map = {uid: Adam(weights=u, lr=init_lr) for uid, u in self.user_map.items()}
        # self.history.add(time.time() - self.start_time, self.RMS(self.test_data))
        for ste in range(epoch):
            for (uid, iid, r) in self.train_data:
                u_vec = self.user_map[uid]
                v_vec = self.item_map[iid]
                if not fake_foreign and iid in self.foreign:
                    continue
                e = (float(r) / 5) - np.dot(u_vec, v_vec.T)
                vg = (e * u_vec)
                ug = (e * v_vec)
                self.user_map[uid] += init_lr * (ug - lambda_1 * u_vec)
                self.item_map[iid] += init_lr * (vg - lambda_2 * v_vec)
                # v_adams_map[iid].minimize_raw(-vg + lambda_2 * v_vec)
                # u_adams_map[uid].minimize_raw(-ug + lambda_1 * u_vec)
            rms = self.RMS(self.test_data)
            self.history.add(time.time() - self.start_time, rms)
            if self.verbose > 0:
                print('{}||epoch{},rms={}'.format(self.name, ste, rms))
        total_gradients_v = {iid: (self.apply_dp(iid, self.item_map[iid] - v_old[iid])) for iid in self.item_map.keys()}
        if self.parent.aggregator.asy and queue:
            queue.put((self.name, total_gradients_v))
        return total_gradients_v
