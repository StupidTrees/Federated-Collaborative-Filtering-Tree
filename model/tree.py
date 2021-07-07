class Tree:
    def __init__(self,name):
        self.name = name
        self.root = None
        self.layers = []
        self.leaves = []

    def get_at(self, layer, index):
        return self.layers[layer][index]
