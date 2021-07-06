class Optimizer:
    def __init__(self, init_lr=0.06, lr_decay_rate=0.002):
        self.lr_decay_rate = lr_decay_rate
        self.init_lr = init_lr
        self.node = None
        self.lr = 0.0
        self.activate_time = 0

    def reset(self):
        pass

    def round_begin(self, init_lr):
        self.lr = init_lr
        pass

    def get_self_lr(self, init_lr, epoch, parent_epoch):
        """
        leaf的optimizer调用，在某轮训练中指定自己的学习率
        :param init_lr: 自己的初始学习率
        :param epoch: 哪一轮
        :param parent_epoch: 此时父节点位于哪一轮
        :return: 学习率
        """
        self.activate_time += 1
        self.lr -= self.lr * self.lr_decay_rate
        return self.lr

    def get_child_init_lr(self, init_lr, epoch, child):
        """
        root的optimizer调用，在某轮训练中为某个孩子分配初始学习率
        :param init_lr: root自己的初始学习率
        :param epoch: root当前的epoch
        :param child: 孩子
        :return: 学习率
        """
        self.lr -= self.lr * self.lr_decay_rate
        return self.lr
