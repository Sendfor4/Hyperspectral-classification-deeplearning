# -*- coding: utf-8 -*-
"""
大概是主要的预处理方法都在这个部分
"""
import copy
import numpy as np
import pandas as pd


# padding函数，给数据集添加填充
def padding(dataset, patch_size):
    # 使用常数0在周围进行padding情况
    # padding_dataset = copy.deepcopy(dataset)
    # p = patch_size // 2  # 补丁一半的大小
    # # 对数据进行填充，'constant'指定填充类型为常数，constant_values = 0表示填充值为0
    # padding_dataset = np.pad(padding_dataset, ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)
    # return padding_dataset

    # 在数据集的上下分别加patch/2行数据(最近的那一行重复加)，再在左右分别加patch/2列数据
    padding_dataset = copy.deepcopy(dataset)
    pad = patch_size//2
    up_row = np.repeat([padding_dataset[0, :, :]], pad, axis=0)
    down_row = np.repeat([padding_dataset[-1, :, :]], pad, axis=0)
    padding_dataset = np.concatenate((up_row, padding_dataset, down_row), axis=0)
    # 按列重复完要转置一下，不然拼接不上
    left_column = np.repeat([padding_dataset[:, 0, :]], pad, axis=0).transpose((1, 0, 2))
    right_column = np.repeat([padding_dataset[:, -1, :]], pad, axis=0).transpose((1, 0, 2))
    padding_dataset = np.concatenate((left_column, padding_dataset, right_column), axis=1)
    return padding_dataset


# 制作训练集切片,构建3D-patch
def make_train_patch(dataset, coordinate, patch_size):
    current_class_train_sample = list()
    for row in coordinate:
        current_class_train_sample.append(dataset[row[0]:row[0]+patch_size, row[1]:row[1]+patch_size, :])
    return np.array(current_class_train_sample)


# 制作训练集
def make_train_set(dataset, ground_truth, sample_num_per_class):
    percent_flag = False  # 此时表面不按照比例取数据集
    if sample_num_per_class >= 1.0:  # sample_num_per_class > 1 表明每类像素都按具体的数量取，而不是比例
        sample_num_per_class = int(sample_num_per_class)
    else:
        percent_flag = True

    predict_ground_truth = copy.deepcopy(ground_truth)

    # 搜寻每类标签的坐标,随机打乱，取每类前sample_num_per_class个作为训练集
    patch_size = dataset.shape[0] - ground_truth.shape[0] + 1
    label_coordinate = list()


# 这个类是用来输出详细数据表格的详细数据包括什么训练时间、OA、F1score之类的，还有记录argument里面的一些重要参数。
class OutputData:
    def __init__(self, num_of_label, trial_times):
        self.trial_times = trial_times
        self.params = ('AA', 'OA', 'micro_F1_score', 'train_time', 'predict_time')
        self.params_class = tuple(np.arange(1, num_of_label + 1))
        self.args = ('dataset', 'method', 'patch', 'samples_per_class')
        self.chart = pd.DataFrame(np.zeros((trial_times + 2, len(self.params_class + self.params))),
                                  index=np.arange(1, trial_times + 3), columns=self.params_class+self.params)
        self.chart = self.chart.rename(index={trial_times + 1: 'average'})
        self.chart = self.chart.rename(index={trial_times + 2: 'arguments'})

    def set_data(self, param_name, current_trail_turn, data):
        self.chart[param_name][current_trail_turn + 1] = data

    # 算一下均值，把argument里面的一些参数写在最后一行，然后就输出
    def output_data(self, path, arguments):
        file_name = path + 'detail_data.xlsx'
        for i in self.params_class:
            self.chart[i]['average'] = self.chart[i][0:self.trial_times].mean()
        for param_name in self.params:
            self.chart[param_name]['average'] = self.chart[param_name][0:self.trial_times].mean()
        for i, arg_name in enumerate(self.args):
            self.chart[self.params[i]]['arguments'] = arg_name + ":" + str(arguments[arg_name])
        self.chart.to_excel(file_name, 'detail data')