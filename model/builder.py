from model.tree import Tree
from train.data import split_data
from model.leaf import Leaf
from model.root import Root


def build_tree_horizontal(full_data_path, child_sizes, tree_sizes, k=10, name_prefix=''):
    """
    生成一棵简单的横向联邦学习树
    :param name_prefix: 命名前缀
    :param full_data_path:原始数据文件
    :param child_sizes:叶子层的各个叶子包含的用户数量表，从左到右表示
    :param tree_sizes: 叶子层往上的各层的形状，例如[[1,2,1],[2,1],[2]]
        表示一棵这样的树：
                        [root]
                        /   \
                  [root]    [root]
                  /  \          \
             [root]  [root]   [root]
            /        /  \     / \
        [leaf]  [leaf] [leaf]  [leaf]
    :param k: 隐含相关度
    :return: 逐层表示的数结构，是一个结点列表，每一项代表一层
    """
    data = split_data(full_data_path, size_list=child_sizes, horizontal=True, output=False)
    total_nodes = []
    children = []
    for idx, tp in enumerate(data):
        children.append(Leaf('{}0.{}'.format(name_prefix, idx), data_tuple=tp, k=k))
    for height, layer in enumerate(tree_sizes):
        total_nodes.append(children)
        left = 0
        nodes = []
        for idx, sz in enumerate(layer):
            nodes.append(Root('{}{}.{}'.format(name_prefix, height + 1, idx), k=k, clients=children[left:left + sz]))
            left += sz
        children = nodes
    total_nodes.append(children)
    result = Tree(name=name_prefix)
    result.layers = total_nodes
    result.root = total_nodes[-1][0]
    result.leaves = total_nodes[0]
    return result

# layers = build_tree_horizontal('../data/full', [10, 20, 30, 40, 50], [[2, 2, 1], [1, 2], [2]])
