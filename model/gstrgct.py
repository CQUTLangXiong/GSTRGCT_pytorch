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
from scipy.sparse.linalg import eigs
from .tgct_encoder import my_Layernorm, EncoderLayer, Encoder
from .embed import PositionalEmbedding
from .auto_correlation import AutoCorrelation, AutoCorrelationLayer
# from .auto_correlation_org import AutoCorrelation, AutoCorrelationLayer


root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
def transform_numpy_to_tensor(data, device=device):
    data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    return data
def scaled_Laplacian(W):
    '''
    W是tensor
    :return: #计算标准化拉普拉斯矩阵
    '''
    assert W.shape[0] == W.shape[1]  # 确保是方阵

    # D = np.diag(np.sum(W, axis=1))  # 计算度
    D = torch.diag(torch.sum(W, dim=1)).to(device)

    L = D - W
    # scipy.sparse.linalg.eigs(A,k,M,sigma,which='',..)求稀疏矩阵A的k个特征值和特征向量
    # lambda_max = eigs(L, k=1, which='LR')[0].real  # 最大的实数部分特征值
    lambda_max = max([torch.real(i) for i in torch.linalg.eigvals(L)]).to(device)

    return (2 * L) / lambda_max - torch.eye(W.shape[0]).to(device)  # 波浪符号的L：切比雪夫多项式的自变量


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N),若是tensor用下面的

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    # cheb_polynomials = [np.identity(N), L_tilde.copy()]  # 0阶，1阶
    cheb_polynomials = [torch.eye(N), L_tilde.repeat(1,1)]# 0阶，1阶

    for i in range(2, K):  # 2阶，k-1阶
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

class SWNN(nn.Module):
    """
    compute semantic spatial weight
    """
    def __init__(self, num_of_vertices, n_hidden):
        super(SWNN, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.n_hidden = n_hidden
        self.linear = nn.Linear(self.num_of_vertices, self.num_of_vertices)
    def forward(self, W):
        for hidden in range(self.n_hidden):
            W = self.linear(W)
            W = F.softmax(W, dim=1)
        return W

# 空间面
class SRGCN(nn.Module):
    def __init__(self, num_of_vertices, K,  in_channels, out_channels, reg_mode="gsr"): # 1层
        """
        :param K:
        :param cheb_polynomials: 切比雪夫多项式
        :param in_channels: 输入特征数
        :param spatial_channels:
        :param out_channels: 输出特征数
        """
        super(SRGCN, self).__init__()
        self.swnn = SWNN(num_of_vertices, n_hidden=1).to(device)  #用来对权重进行计算
        self.K = K
        # self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reg_mode = reg_mode

        # 共享
        self.alpha = nn.ParameterList([nn.Parameter(torch.FloatTensor(1).to(device)) for _ in range(K)])
        self.beta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(device)) for _ in range(K)])
        self.theta= nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(device)) for _ in range(K)])
    def forward(self, s_w, x) :# x-->batchsize
        W = self.swnn(s_w).to(device)
        batch_size, num_time_steps, num_of_vertices, in_channels = x.shape #
        assert W.shape[0] == W.shape[1]  # 确保是方阵
        assert W.shape[1] == num_of_vertices  # 确保W的维度与x的节点数一致
        #  计算标准化拉普拉斯矩阵L_tilde
        L_tilde = scaled_Laplacian(W).to(device)
        cheb_polynomials = cheb_polynomial(L_tilde, self.K) #列表
        outputs = []
        for time_step in range(num_time_steps):
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(device)
            l_N = torch.ones((batch_size, num_of_vertices,1)).to(device)
            x_t = x[:, time_step, :, :] # [batch_size, num_of_vertices, in_channels]
            w_x_t = x_t.permute(0,2,1).matmul(W).permute(0,2,1) # [batch_size, num_of_vertices, in_channels]
            for k in range(self.K):
                T_k = cheb_polynomials[k].to(device) # 取出切比雪夫多项式第k项
                alpha_k = self.alpha[k]
                beta_k = self.beta[k]
                theta_k = self.theta[k]

                # rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                if self.reg_mode=="gsr":
                    s_r = alpha_k * l_N + x_t.matmul(beta_k) + w_x_t.matmul(theta_k) #空间回归
                    s_r = s_r.to(device)
                    output += s_r.permute(0, 2, 1).matmul(T_k).permute(0, 2,1)  # [batch_size, num_of_vertices, out_channels]
                elif self.reg_mode=="normal":
                    s_r = x_t.matmul(beta_k)
                    s_r = s_r.to(device)
                    output += s_r.permute(0, 2, 1).matmul(T_k).permute(0, 2,1)  # [batch_size, num_of_vertices, out_channels]
                else:
                    print("模型输入错误")
                # output += s_r.permute(0,2,1).matmul(T_k).permute(0,2,1) # [batch_size, num_of_vertices, out_channels]
            outputs.append(output.unsqueeze(-1)) # [batch_size, num_of_vertices, out_channels,1]
        # # [batch_size, num_of_vertices, out_channels,num_time_steps]
        return F.relu(torch.cat(outputs, dim=-1))



class AutoTRGCN(nn.Module):
    def __init__(self, num_time_steps_in, num_time_steps_out, d_model, n_heads, dropout, num_layers): #
        super(AutoTRGCN, self).__init__()
        # self.num_of_vertices = num_of_vertices
        self.num_time_steps_in = num_time_steps_in
        self.num_time_steps_out =num_time_steps_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.pos_enc = PositionalEmbedding(self.d_model)
        self.auto_correlation = AutoCorrelation(mask_flag=True, factor=1, scale=None, attention_dropout=0.1,output_attention=True)
        self.auto_correlation_layer = AutoCorrelationLayer(self.auto_correlation, self.d_model, self.n_heads, d_keys=None, d_values=None)
        self.encoder_layer = EncoderLayer(self.auto_correlation_layer, self.d_model, d_ff=None, dropout=self.dropout, activation="relu")
        self.encoder_layers = [self.encoder_layer for l in range(num_layers)]
        self.encoder = Encoder(self.encoder_layers, conv_layers=None, norm_layer=my_Layernorm(self.d_model))
        self.decoder = nn.Linear(d_model*num_time_steps_in, num_time_steps_out)

    def forward(self, x):
        pe_x = self.pos_enc(x)   #
        # print(pe_x.shape)
        encoder_out, attns = self.encoder(pe_x)     # [batch_size, num_of_vertices, in_channels, num_time_steps_in_put, d_model]
        t_out = F.relu(self.decoder(torch.flatten(encoder_out,-2,-1)))    # [batch_size, num_of_vertices, in_channels, num_time_steps_input]
        # pe_x = self.pos_enc(x)        # [batch_size, num_of_vertices, in_channels, num_time_steps_in_put, d_model]
        # pe_x = pe_x.permute(1,2,0,3,4)   # [num_of_vertices, in_channels,batch_size,num_time_steps_in_put, d_model]
        # num_of_vertices =  pe_x.shape[0]
        # #   节点共享参数，因为研究的是时序特征
        # mean_value =  torch.mean(pe_x, dim=0)   #   in_channels,batch_size,num_time_steps_in_put, d_model
        # c_out = []
        # # 单个特征的时序特征        ，预测一定是根据将来的特征计算将来的值过去的值只能说明趋势性，不能起决定作用，这才是预测的本质
        # # 做到预测将来特征的时序值
        # for  i in range(mean_value.shape[0]):
        #     c_pe_x = mean_value[i, :, :, :]      # [batch_size, num_time_steps_in, d_model]
        #     out, _= self.encoder(c_pe_x)
        #     c_out.append(out.unsqueeze(0))
        # c_out = torch.cat(c_out, dim=0)          # [in_channel, batch_size, num_time_steps_in, d_model]
        # n_out = c_out.unsqueeze(0).repeat(num_of_vertices,1,1,1,1) # [num_of_vertices, in_channels,batch_size,num_time_steps_in_put, d_model]
        # n_out = n_out.permute(2,0,1,3,4)  # 恢复输入的格式  [batch_size, num_of_vertices, in_channels, num_time_steps_in_put, d_model]
        # # decoder
        # t_out = F.relu(self.decoder(torch.flatten(n_out,-2,-1)))
        return t_out, attns
class BaseTRGCN(nn.Module):
    def __init__(self, num_time_steps_in, num_time_steps_out, d_model, n_heads, dropout, num_layers):
        super(BaseTRGCN, self).__init__()
        self.num_time_steps_in = num_time_steps_in
        self.num_time_steps_out =num_time_steps_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.pos_enc = PositionalEmbedding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                        nhead=self.n_heads,
                                                        dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model*num_time_steps_in, num_time_steps_out)
    def forward(self, x):
        pe_x = self.pos_enc(x)        # [batch_size, num_of_vertices, in_channels, num_time_steps_in_put, d_model]
        pe_x = pe_x.permute(1,2,0,3,4)   # [num_of_vertices, in_channels,batch_size,num_time_steps_in_put, d_model]
        num_of_vertices =  pe_x.shape[0]
        #   节点共享参数，因为研究的是时序特征
        mean_value =  torch.mean(pe_x, dim=0)   #   in_channels,batch_size,num_time_steps_in_put, d_model
        c_out = []
        # 单个特征的时序特征        ，预测一定是根据将来的特征计算将来的值过去的值只能说明趋势性，不能起决定作用，这才是预测的本质
        # 做到预测将来特征的时序值
        for  i in range(mean_value.shape[0]):
            c_pe_x = mean_value[i, :, :, :]      # [batch_size, num_time_steps_in, d_model]
            out = self.transformer_encoder(c_pe_x)
            c_out.append(out.unsqueeze(0))
        c_out = torch.cat(c_out, dim=0)          # [in_channel, batch_size, num_time_steps_in, d_model]
        n_out = c_out.unsqueeze(0).repeat(num_of_vertices,1,1,1,1) # [num_of_vertices, in_channels,batch_size,num_time_steps_in_put, d_model]
        n_out = n_out.permute(2,0,1,3,4)  # 恢复输入的格式  [batch_size, num_of_vertices, in_channels, num_time_steps_in_put, d_model]
        # decoder
        t_out = F.relu(self.decoder(torch.flatten(n_out,-2,-1)))
        return t_out








class GSTRGCT(nn.Module):
    def __init__(self, num_of_vertices,in_channels, out_channels, num_time_steps_in, num_time_steps_out,
                 dropout=0.1, K=1, d_model=64, n_heads=8, num_layers=2,
                 reg_mode="gsr", type = 'auto', tensor_deco=True):
        super(GSTRGCT, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self.out_channels  = out_channels
        self.num_time_steps_in = num_time_steps_in
        self.num_time_steps_out = num_time_steps_out
        self.reg_mode = reg_mode
        self.type = type
        self.tensor_deco=tensor_deco
        self.srgcn = SRGCN(self.num_of_vertices, K,  self.in_channels, self.out_channels, self.reg_mode).to(device)
        self.auto_trgcn = AutoTRGCN(self.num_time_steps_in, self.num_time_steps_out, d_model, n_heads, dropout, num_layers)
        self.base_trgcn = BaseTRGCN(self.num_time_steps_in, self.num_time_steps_out, d_model, n_heads, dropout, num_layers)

        self.s_linear = nn.Linear(self.num_time_steps_in, self.num_time_steps_out)
        self.t_linear = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, s_w, x): # [batch_size, num_time_steps_in, num_of_vertices, in_channels]  [64,12, 307,3]
        if self.tensor_deco:
            s_out = self.srgcn(s_w, x)      # [batch_size,  num_of_vertices, out_channels, num_time_steps_in]
            s_out = self.s_linear(s_out)    #  [batch_size,  num_of_vertices, out_channels, num_time_steps_out]
            if self.type=="auto":
                t_out, attns = self.auto_trgcn(x.permute(0,2,3,1))     # [batch_size, num_of_vertices, in_channels, num_time_steps_out]
                # t_out= self.auto_trgcn(x.permute(0, 2, 3, 1))
                t_out = self.t_linear(t_out.permute(0, 1, 3, 2))             # [batch_size, num_of_vertices, num_time_steps_out,out_channels]
                t_out = t_out.permute(0, 1, 3, 2)     # [batch_size,  num_of_vertices, out_channels, num_time_steps_out]
            elif self.type=="base":     # base
                t_out = self.base_trgcn(x.permute(0,2,3,1))
                t_out = self.t_linear(t_out.permute(0, 1, 3, 2))
                t_out = t_out.permute(0, 1, 3, 2)
            else:
                print("模型输入错误，请检查")
            st_out = F.relu(s_out*t_out)                       #    哈达马积    [batch_size,  num_of_vertices, out_channels, num_time_steps_out]
            # st_out = s_out * t_out
        else:
            s_out = self.srgcn(s_w, x)
            if self.type == "auto":
                t_out, attns = self.auto_trgcn(s_out)   # [batch_size, num_of_vertices, out_channels, num_time_steps_out]
                # t_out = self.auto_trgcn(s_out)
            elif self.type == "base":
                t_out = self.base_trgcn(s_out)
            else:
                print("模型输入错误，请检查")
            st_out = t_out
                # st_out = F.relu(s_out+t_out)
        return st_out.permute(0,3,1,2)      #  [batch_size,  num_time_steps_out,num_of_vertices, out_channels]  [64,3,307,1]




if __name__ == '__main__':
    x = torch.rand(64, 12, 170, 3).to(device)
    s_w = torch.rand(170,170).to(device)
    # auto_trgcn = AutoTRGCN(num_time_steps_in=12, num_time_steps_out=3, d_model=64, n_heads=8, dropout=0.1, num_layers=1).to(device)
    base_trgcn = BaseTRGCN(num_time_steps_in=12, num_time_steps_out=3, d_model=64, n_heads=8, dropout=0.1, num_layers=1).to(device)
    out = base_trgcn(x.permute(0,2,3,1))
    print(out.shape)
    gstrgct = GSTRGCT(num_of_vertices=170,in_channels=3, out_channels=1, num_time_steps_in=12, num_time_steps_out=3,
                      d_model=64, n_heads=8, num_layers=1,
                      reg_mode="gsr", type = 'auto', tensor_deco=True).to(device)
    st_out = gstrgct(s_w,x)
    print(st_out.shape)























# if __name__ == "__main__":
#     # A, A_distance, dist_mean, dist_std=get_adjacency_matrix(root_data_path + '/pems08/distance.csv', 307)
#     # print(A.shape, distanceA.shape)
#     # dp = DataProcess(root_data_path, data_file_name=root_data_path + '/pems04/pems04.npz', type='pems04')
#     # data, scale_data, _min, _max= dp.load_data()
#     # all_features, all_target =dp.generate_dataset(scale_data, num_time_steps_input=12, num_time_steps_output=3, role="traffic flow prediction")
#     # all_data = dp.get_train_valid_data(all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2)
#     # all_data_loader = dp.get_data_loader(all_data, save=True)
#     wp = WeightProcess(adj_filename='/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data/pems04/distance.csv', num_of_vertices=307, type='pems04')
#     A, A_distance, dist_mean, dist_std = wp.get_adjacency_matrix()
#     s_w = wp.get_smooth_weight_matrix(A, A_distance, dist_mean, dist_std, scaling=True)
#     s_w = torch.from_numpy(s_w).to(device)
#
#     # swnn = SWNN(num_of_vertices=307, n_hidden=1)
#     x = torch .rand(64, 12, 307, 3).to(device)
#     srgcn = SRGCN(num_of_vertices=307, K=1,  in_channels=3, out_channels=1).to(device)
#     output = srgcn(s_w, x)
#     print(output.shape)
#
#
#
#     # L_tilde = scaled_Laplacian(W)
#     # cheb = cheb_polynomial(L_tilde, 1)
#     # print(L_tilde.shape)
#     # print(L_tilde)
#     # print(cheb[0].shape)


    # W = weight_matrix(A, A_distance, dist_mean, dist_std, scaling=True)
    # weight_plot(W)
    # plot_data(data, time_index=24)
    # print(W)
    # print(data.shape)
    # # print(scale_data)
    # print(scale_data.shape)
    # print(_min.shape)
    # print(_max.shape)
    # print(all_features.shape)
    # print(all_target.shape)

    # features, target = generate_dataset(scale_data, num_timesteps_input=12, num_timesteps_output=3)
    # print(features.shape)
    # print(target.shape)






