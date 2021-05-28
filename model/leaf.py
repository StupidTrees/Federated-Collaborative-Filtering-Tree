import copy
import time

from model.node import *
from train.data import load_data


class Leaf(Node):
    """
    叶子节点，继承自Node
    """

    def __init__(self, name, data_tuple=None, file_name=None, k=10, verbose=0, optimizer=None):
        Node.__init__(self, name, k=k, clients=None, verbose=verbose, optimizer=optimizer)
        if data_tuple is None:
            self.train_data, self.test_data, users, items = load_data(file=file_name,
                                                                      shuffle=True,
                                                                      batch_size=0,
                                                                      tensor=False,
                                                                      test_ratio=0.3)
        else:
            self.train_data, users, items = data_tuple
        for uid in users:
            self.user_map[uid] = np.random.normal(0, 0.1, k)
        for iid in items:
            self.item_map[iid] = np.random.normal(0, 0.1, k)
        self.first_start = True
        self.start_time = time.time()
        self.average_rating = {}
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
        self.first_start = True
        self.optimizer.reset()

    def expand_v(self, iid_list):
        Node.expand_v(self, iid_list)
        for iid in iid_list:
            if iid in self.foreign:
                for uid in self.user_map.keys():
                    self.train_data.append((uid, iid, self.average_rating[uid]))

    def RMS(self, batch):
        sum = 0.0
        count = len(batch)
        for (uid, iid, r) in batch:
            if uid in self.user_map.keys() and iid in self.item_map.keys():
                pr = self.predict(uid, iid)
                sum += np.square((float(r) / 5) - pr)
        if count == 0:
            return 0
        return np.sqrt(sum / count)

    def update_v(self, v_map):
        for iid in self.item_map.keys():
            if iid in v_map.keys():
                self.item_map[iid] = v_map[iid]

    def apply_dp(self, v_vec):
        """
        对v向量进行加噪处理，加入拉普拉斯分布的随机噪声使其满足epsilon-差分隐私
        :param v_vec: v向量
        :return: 差分后的
        """
        return v_vec + np.random.laplace(0, 0.1 / 0.5, 10)

    def do_train(self, epoch=10, parent_ste=0, init_lr=0.005, asy=True,
                 trans_delay=0.5, lambda_1=0.1, lambda_2=0.1, queue=None, fake_foreign=False):
        if self.first_start:
            self.start_time = time.time()
            self.first_start = False
        v_old = copy.deepcopy(self.item_map)
        self.optimizer.round_begin(init_lr=init_lr)
        for ste in range(epoch):
            tmp_grad_u = {}
            tmp_grad_v = {}
            lr = self.optimizer.get_self_lr(init_lr, ste, parent_ste)
            for (uid, iid, r) in self.train_data:
                u_vec = self.user_map[uid]
                v_vec = self.item_map[iid]
                if not fake_foreign and iid in self.foreign:
                    continue
                e = (float(r) / 5) - np.dot(u_vec, v_vec.T)
                vg = (e * u_vec)
                ug = (e * v_vec)
                self.user_map[uid] += lr * (ug - lambda_1 * u_vec)
                self.item_map[iid] += lr * (vg - lambda_2 * v_vec)
                if iid not in tmp_grad_v.keys():
                    tmp_grad_v[iid] = 0.0
                if uid not in tmp_grad_u.keys():
                    tmp_grad_u[uid] = 0.0
                tmp_grad_u[uid] += ug
                tmp_grad_v[iid] += vg
            rms = self.RMS(self.test_data)
            self.history.add(time.time() - self.start_time, rms)
            if self.verbose > 0:
                print('{}||epoch{},rms={}'.format(self.name, ste, rms))
        total_gradients_v = {iid: (self.apply_dp(self.item_map[iid] - v_old[iid])) for iid in self.item_map.keys()}
        if asy and queue:
            queue.put((self.name, total_gradients_v))
        return total_gradients_v
