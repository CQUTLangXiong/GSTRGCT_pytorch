import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from metrics import masked_mape_np
from scipy.sparse.linalg import eigs
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
fig_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/fig'
root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
def transform_numpy_to_tensor(data, device=device):
   data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
   return data






def plot_data(data, time_index=0):
    data = data[time_index,:,:].reshape(-1,3)
    scale = MinMaxScaler()
    data = scale.fit_transform(data)
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # ax = Axes3D(fig)
    # ax = plt.subplot(projection='3d')

    ax.scatter(data[:,0],data[:,1],data[:,2])
    plt.savefig(fig_path+'/data_%s.png' % time_index,dpi=300,bbox_inches='tight')

# A_distance平滑得到权重


def weight_plot(W):
    ax = sns.heatmap(W)
    ax.set_xlabel('Station ID')
    ax.set_ylabel('Station ID')
    plt.savefig(fig_path +'/W.png', dpi=300, bbox_inches='tight')

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]          # 0阶，1阶

    for i in range(2, K):   #2阶，k-1阶
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials















# if __name__ == "__main__":
    # A, A_distance, dist_mean, dist_std=get_adjacency_matrix(root_data_path + '/pems08/distance.csv', 307)
    # print(A.shape, distanceA.shape)
    # dp = DataProcess(root_data_path, data_file_name=root_data_path + '/pems04/pems04.npz', type='pems04')
    # data, scale_data, _min, _max = dp.load_data()
    # all_features, all_target = dp.generate_dataset(scale_data, num_time_steps_input=12, num_time_steps_output=3,
    #                                                role="traffic flow prediction")
    # all_data = dp.get_train_valid_data(all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2)
    # all_data_loader = dp.get_data_loader(all_data, save=True)

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




