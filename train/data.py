import random

import matplotlib.pyplot as plt
import tensorflow as tf
import os


def split_data(raw, size_list=None, horizontal=True, output=True, save_path='data/'):
    """
    切分数据
    :param raw:原始文件
    :param size_list: 指定切分后的大小，比如[10,20,30]
    :param horizontal: 是否横向（按用户）切分
    :param output: 是否输出到文件
    :param save_path: 输出文件的path
    :return: 切分结果
    """
    user_set = {}
    item_set = {}
    u_idx = 0
    i_idx = 0
    f = open(raw)
    data = []
    outF = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ln = 0
    for line in f.readlines():
        u, i, r, t = line.split()
        if int(u) not in user_set:
            user_set[int(u)] = u_idx
            u_idx += 1
        if int(i) not in item_set:
            item_set[int(i)] = i_idx
            i_idx += 1
        if horizontal:
            x_index = user_set[int(u)]
            mark = 'U'
        else:
            x_index = item_set[int(i)]
            mark = 'I'
        file_index = 0
        sum = 0
        for sz in size_list:
            sum += sz
            if x_index < sum:
                break
            file_index += 1
        if file_index >= len(data):
            if file_index >= len(size_list):
                continue
            data.append(([], set(), set()))
            if output:
                outF.append(open('{}{}_{}{}'.format(save_path, f.name.split('/')[-1], mark, file_index + 1), 'w+'))
        data[file_index][1].add(int(u))
        data[file_index][2].add(int(i))
        data[file_index][0].append([int(u), int(i), int(r)])
        if output:
            outF[file_index].write(line)
        ln += 1
    for dt, _, _ in data:
        random.shuffle(dt)
    if output:
        for of in outF:
            of.close()
    f.close()
    return data


def load_data(file, batch_size=32, shuffle=True, tensor=True, test_ratio=0.2, uid_range=None, iid_range=None):
    """
    加载数据集
    :param file:数据集文件
    :param batch_size: mini-batch的大小
    :param shuffle: 是否打乱
    :param tensor: 是否以张量形式返回
    :param test_ratio: 取多少比率作为测试集
    :param uid_range: 指定只加载这些范围内的用户
    :param iid_range: 指定只加载这些范围内的物品
    :return:
    """
    user_map = set()
    item_map = set()
    data = []
    f = open(file)
    for line in f.readlines():
        u, i, r, t = line.split()
        if uid_range is not None and int(u) not in uid_range:
            continue
        if iid_range is not None and int(i) not in iid_range:
            continue
        if int(u) not in user_map:
            user_map.add(int(u))
        if int(i) not in item_map:
            item_map.add(int(i))
        data.append([int(u), int(i), int(r)])  # user_map[int(u)], item_map[int(i)], int(r)])
    f.close()
    if tensor:
        ds = tf.data.Dataset.from_tensor_slices(data).prefetch(10)
        if batch_size > 0:
            ds = ds.batch(batch_size)
        if shuffle:
            ds = ds.shuffle(1024)
        test_num = int(len(ds) * test_ratio)
        return ds.skip(test_num), ds.take(test_num).unbatch(), user_map, item_map
    else:
        if shuffle:
            random.shuffle(data)
        return data[0:int(len(data) * (1 - test_ratio))], data[int(len(data) * (1 - test_ratio)):], user_map, item_map
