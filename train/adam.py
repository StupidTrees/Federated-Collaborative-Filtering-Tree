import numpy as np


class Adam:
    def __init__(self, weights, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.theta = weights
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.m_cat = 0
        self.v_cat = 0
        self.v = 0
        self.t = 0

    def minimize_raw(self, gradient):
        self.t += 1
        g = gradient
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m_cat = self.m / (1 - self.beta1 ** self.t)
        self.v_cat = self.v / (1 - self.beta2 ** self.t)
        self.theta -= self.lr * self.m_cat / (self.v_cat ** 0.5 + self.epsilon)

    def minimize(self, gradient):
        self.t += 1
        g = gradient
        lr = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.theta -= lr * self.m / (self.v ** 0.5 + self.epsilon)
