from typing import Dict, Union, Any
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import json
import matplotlib.pyplot as plt
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from tensorboardX import SummaryWriter
import os
from pprint import pprint
from lib.utils import transform_numpy_to_tensor

# 设置seed
my_seed = 20220925
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

root_data_path =  '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
class DataProcess:
    def __init__(self, root_data_path , data_file_name, batch_size, type="pems04"):
        """"
        param: data_file_name-->pems04.npz/pems08.npz绝对路径,shape(16992, 307, 3),(17856, 170, 3)
        """
        self.root_data_path = root_data_path
        self.data_file_name = data_file_name
        self.batch_size = batch_size
        self.type = type

    def max_min_normalization(self, x, _max, _min):
        x = 1. * (x - _min) / (_max - _min)
        x = x * 2. - 1.
        return x

    def re_max_min_normalization(self, x, _max, _min):
        x = 1. * x * (_max - _min) + _min
        return x

    def load_data(self): #暂时不变化
        scale = MinMaxScaler()
        scale_data = []
        _min = []
        _max = []
        data = np.load(self.data_file_name)
        data = data['data']
        # 将数据进行标准化
        for i in range(data.shape[0]):
            scale_data.append(scale.fit_transform(data[i, :, :].reshape(-1, 3)))
            _min.append(data[i, :, :].reshape(-1, 3).min(axis=0).reshape(-1,3))
            _max.append(data[i, :, :].reshape(-1, 3).max(axis=0).reshape(-1,3))

        return data, np.array(scale_data), np.array(_min), np.array(_max)
    def generate_dataset(self, scale_data, num_time_steps_input, num_time_steps_output, role="traffic flow prediction"):
        """
        :param scale_data: # 标准化的数据[0,1]
        :param num_time_steps_input:输入时间步数
        :param num_time_steps_output: 预测时间步数
        :param role: “模型用作交通流预测”，否则应该重新生成样本集
        :return: 生成样本集X,Y-->tensor张量，
        """
        X = scale_data #维度为（总的时间序列长度, 节点数, 特征数） #特征数也叫通道数，在统计学中3个通道数实际是独立的特征，因此标准化没有问题
        features, target = [],[]
        if role == "traffic flow prediction":
            indices = [(i, i + (num_time_steps_input + num_time_steps_output)) for i in range(X.shape[0] - (num_time_steps_input + num_time_steps_output) + 1)] #产生样本集的其实索引与终止y的索引
            # 注意i:i+ num_timesteps_input为x序列索引，i+ num_timesteps_input:(i + num_time_steps_input) + num_time_steps_output为预测y的索引

            # 取出序列索引(0,0+num_timesteps_input + num_timesteps_output),(1,1+num_timesteps_input + num_timesteps_output)
            # (34272-num_timesteps_input + num_timesteps_output),34272)
            # 长度为34272-num_timesteps_input + num_timesteps_output+1

            # Save samples
            for i, j in indices:
                features.append(
                    X[i: i + num_time_steps_input, :, :])  # 取出第i个步长为num_timesteps_input的序列
                target.append(X[i + num_time_steps_input: j, :, 0].reshape(num_time_steps_output,-1,1))  # 为什么是0，目标是流量,对于多维时间序列预测，预测的特征，在过去也是输入的特征,reshape 保证X,y维度个数一致
                ##取出第i个预测步长为num_timesteps_output的序列，维度为（207，num_timesteps_input，2）
        else:
            print("该模型不是用于交通流预测，请重新生成样本集")

        return np.array(features), np.array(target) # X:维度为(样本个数(序列个数）,输入时间步数，节点数，特征数）Y:维度为(样本个数(序列个数）,输出时间步数，节点数，特征数=1）
    def get_train_valid_data(self, all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2):
        """
        :param all_features: X样本集,维度为(样本个数(序列个数）,输入时间步数，节点数，特征数
        :param all_target: Y样本集,维度为(样本个数(序列个数）,输出时间步数，节点数，特征数=1）
        :param split_size_1: 划分训练集的样本比例
        :param split_size_2: 划分测试集的样本比例
        :return: all_data-->json
        """
        split_1 = int(len(all_features)*split_size_1)
        split_2 = int(len(all_features)*(split_size_1 +split_size_2))

        train_x = all_features[:split_1, :, :, :]
        val_x = all_features[split_1:split_2, :, :, :]
        test_x = all_features[split_2:, :, :, :]

        train_y = all_target[:split_1, :, :, :]
        val_y = all_target[split_1:split_2, :, :, :]
        test_y = all_target[split_2:, :, :, :]
        all_data = {
            'train': {
                'x': train_x,
                'y': train_y
            },
            'val': {
                'x': val_x,
                'y': val_y
            },
            'test': {
                'x': test_x,
                'y': test_y
            },
            'stats': {
                'min': _min,
                'max': _max
            }
        }
        return all_data

    def get_data_loader(self, all_data, save=True):
        train_x = all_data['train']['x']
        train_y = all_data['train']['y']

        val_x = all_data['val']['x']
        val_y = all_data['val']['y']

        test_x = all_data['test']['x']
        test_y = all_data['test']['y']

        _min = all_data['stats']['min']
        _max = all_data['stats']['max']

        # ------train_loader------
        train_x_tensor = transform_numpy_to_tensor(train_x)
        train_y_tensor = transform_numpy_to_tensor(train_y)
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------val_loader------vals[0].real
        val_x_tensor = transform_numpy_to_tensor(val_x)
        val_y_tensor = transform_numpy_to_tensor(val_y)
        val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # ------test_loader------
        test_x_tensor = transform_numpy_to_tensor(test_x)
        test_y_tensor = transform_numpy_to_tensor(test_y)
        test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # ------print size------
        print('train size', train_x_tensor.size(), train_y_tensor.size())
        print('val size', val_x_tensor.size(), val_y_tensor.size())
        print('test size', test_x_tensor.size(), test_y_tensor.size())
        all_data_loader: Dict[str, Union[Dict[str, Any], Dict[str, Any], Dict[str, Union[int, Any]], Dict[str, Any]]] = {
            'train': {
                'x_tensor': train_x_tensor,
                'y_tensor': train_y_tensor,
                'data_loader': train_loader
            },
            'val': {
                'x_tensor': val_x_tensor,
                'y_tensor': val_y_tensor,
                'data_loader': val_loader
            },
            'test': {
                'x_tensor': test_x_tensor,
                'y_tensor': test_y_tensor,
                'data_loader': test_loader
            },
            'stats': {
                'min': _min,
                'max': _max,
                'batch_size': self.batch_size
            }
        }
        if save:
            filename = self.root_data_path + '/%s/train_valid_test_loader_%s.npy' % (self.type, self.batch_size)
            np.save(filename, all_data_loader)
        return all_data_loader


class WeightProcess:
    def __init__(self, adj_file_name, num_of_vertices, type='pems04'):
        self.adj_filename = adj_file_name
        self.num_of_vertices = num_of_vertices
        self.type = type
    def get_adjacency_matrix(self):
        A = np.zeros((int(self.num_of_vertices), int(self.num_of_vertices)), dtype=np.float32)
        A_distance = np.zeros((int(self.num_of_vertices), int(self.num_of_vertices)), dtype=np.float32)

        distance = pd.read_csv(self.adj_filename)
        for i in range(len(distance)):
            from_index = distance['from'][i]
            to_index = distance['to'][i]
            cost = distance['cost'][i]
            A[from_index, to_index] = 1
            A_distance[from_index, to_index] = cost
        dist_mean = distance['cost'].mean()
        dist_std = distance['cost'].std()
        return A, A_distance, dist_mean, dist_std

    def get_smooth_weight_matrix(self, A, A_distance, dist_mean, dist_std, scaling=True):
        """
        :param A_distance: 空间距离矩阵
        :param scaling: 决定是否采用此平滑后的权重矩阵，否则使用0/1的A
        :return:空间权重
        """
        if scaling:
            W = A_distance
            W = np.exp(-(W - dist_mean) * (W - dist_mean) / (dist_std * dist_std)) * A
            # refer to Eq.10
        else:
            W = A # 邻接矩阵
        return W


def get_all_loader(root_data_path, data_file_name, batch_size, type,
                   num_time_steps_input, num_time_steps_output):
    dp = DataProcess(root_data_path, data_file_name, batch_size, type=type)
    data, scale_data, _min, _max = dp.load_data()
    all_features, all_target = dp.generate_dataset(scale_data, num_time_steps_input, num_time_steps_output,
                                                   role="traffic flow prediction")
    all_data = dp.get_train_valid_data(all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2)
    all_data_loader = dp.get_data_loader(all_data, save=False)
    return all_data_loader

if __name__ == '__main__':
    all_data_loader=get_all_loader(root_data_path, data_file_name=root_data_path + '/pems04/pems04.npz',
                                   batch_size=64, type='pems04',
                                   num_time_steps_input=12, num_time_steps_output=3)

