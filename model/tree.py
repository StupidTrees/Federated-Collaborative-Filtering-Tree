from model.coordinator import Coordinator


class Tree:
    def __init__(self, name):
        self.name = name
        self.root = None
        self.layers = []
        self.leaves = []

    def get_at(self, layer, index):
        return self.layers[layer][index]

    def set_grad_max(self, max):
        for node in self.layers[0]:
            node.grad_max = max

    def set_epsilon(self, epsilon):
        for node in self.layers[0]:
            node.epsilon = epsilon

    def set_aggregator(self, agg):
        for layer in self.layers:
            for node in layer:
                if isinstance(node, Coordinator):
                    node.aggregator = agg
                    agg.node = node
