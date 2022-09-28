import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import shutil
import argparse
import configparser
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from model.gstrgct import GSTRGCT
from prepareData import DataProcess, WeightProcess, get_all_loader
from model.astgcn import make_model
from model.lstm import LSTM
from model.stgcn import STGCN
# my_seed = 42
# torch.manual_seed(my_seed)
# torch.cuda.manual_seed_all(my_seed)

root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
config_path = "./configurations/PEMS04_gstrgct.conf"

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss().to(device)
step_size = 3
gamma =0.96


# 预测保存的数据
all_model_val_prediction_truth = {
    'Base-GSTRGCT':{
        'prediction': [],
        'truth': []
    },
    'Auto-GSTRGCT': {
        'prediction': [],
        'truth': []
    },
    'Auto-STRGCT': {
        'prediction': [],
        'truth': []
    },
    'Auto-GSTRGCT_Cas': {
        'prediction': [],
        'truth': []
    },
    'GTWR': {
        'prediction': [],
        'truth': []
    },
    'ARIMA': {
        'prediction': [],
        'truth': []
    },
    'LSTM': {
        'prediction': [],
        'truth': []
    },
    'STGCN': {
        'prediction': [],
        'truth': []
    },
    'ASTGCN': {
        'prediction': [],
        'truth': []
    },
    'SRGNN': {
        'prediction': [],
        'truth': []
    },
}
all_model_test_prediction_truth = {
    'Base-GSTRGCT':{
        'prediction': [],
        'truth': []
    },
    'Auto-GSTRGCT': {
        'prediction': [],
        'truth': []
    },
    'Auto-STRGCT': {
        'prediction': [],
        'truth': []
    },
    'Auto-GSTRGCT_Cas': {
        'prediction': [],
        'truth': []
    },
    'GTWR': {
        'prediction': [],
        'truth': []
    },
    'ARIMA': {
        'prediction': [],
        'truth': []
    },
    'LSTM': {
        'prediction': [],
        'truth': []
    },
    'STGCN': {
        'prediction': [],
        'truth': []
    },
    'ASTGCN': {
        'prediction': [],
        'truth': []
    },
    'SRGNN': {
        'prediction': [],
        'truth': []
    },
}

def get_config():
    # 配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path,
                        type=str, help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    # 读取配置文件
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    return data_config, training_config

# 取配置文件相关参数和变量名
## 数据参数
data_config, training_config = get_config()
data_file_name = data_config['data_file_name']
adj_file_name = data_config['adj_file_name']
num_of_vertices = int(data_config['num_of_vertices'])
dataset_name = data_config['dataset_name']
num_time_steps_input = int(data_config['num_time_steps_input']) #config出来的都是字符，数字需要加int，否则出错
num_time_steps_output = int(data_config['num_time_steps_output'])
in_channels = int(data_config['in_channels'])
out_channels = int(data_config['out_channels'])
## 训练参数
K = int(training_config['K'])
batch_size = int(training_config['batch_size'])
epochs = int(training_config['epochs'])
learning_rate = float(training_config['learning_rate'])
dropout = float(training_config['dropout'])
d_model = int(training_config['d_model'])
n_heads = int(training_config['n_heads'])
num_layers = int(training_config['num_layers'])
start_epoch = 0

# -------实验保存的目录------
exp_dir = os.path.join("experiments", dataset_name) #/experiments/pems04
fig_dir = os.path.join("fig", dataset_name)         # /fig/pems04

# 加载数据
#字典
all_data_loader = get_all_loader(root_data_path, data_file_name,
                                     batch_size, dataset_name,
                                     num_time_steps_input, num_time_steps_output)

train_loader = all_data_loader['train']['data_loader']

val_loader = all_data_loader['val']['data_loader']
val_y_tensor = all_data_loader['val']['y_tensor']

test_loader = all_data_loader['test']['data_loader']
test_y_tensor = all_data_loader['test']['y_tensor']

def evaluate(net, s_w, val_loader):
    net.train(False)
    with torch.no_grad():
        tmp = []
        prediction = torch.Tensor()
        for batch_index, batch_data in enumerate(val_loader):
            x, y = batch_data

            outputs = net(s_w, x)
            loss = criterion(outputs, y)
            tmp.append(loss)
            prediction = torch.cat((prediction, outputs.cpu()))
        validation_loss = sum(tmp) / len(tmp)
    return prediction, validation_loss
def evaluate_astgcn(net, val_loader):
    net.train(False)
    with torch.no_grad():
        tmp = []
        prediction = torch.Tensor()
        for batch_index, batch_data in enumerate(val_loader):
            x, y = batch_data
            x=x.permute(0,2,3,1) # (batch_size, num_of_vertices,in_channels, num_time_steps_input]
            y=y.squeeze(-1).permute(0,2,1) # [batch_size, num_of_vertices, num_time_steps_output]
            outputs = net(x)
            loss = criterion(outputs, y)
            tmp.append(loss)
            prediction = torch.cat((prediction, outputs.cpu()))
        validation_loss = sum(tmp) / len(tmp)
    return prediction, validation_loss
def evaluate_lstm(net, val_loader):
    net.train(False)
    with torch.no_grad():
        tmp = []
        prediction = torch.Tensor()
        for batch_index, batch_data in enumerate(val_loader):
            x, y = batch_data
            outputs = net(x)
            loss = criterion(outputs, y)
            tmp.append(loss)
            prediction = torch.cat((prediction, outputs.cpu()))
        validation_loss = sum(tmp) / len(tmp)
    return prediction, validation_loss

def train(net, s_w, train_loader, val_loader, test_loader, exp_dir, model_name = "Auto-GSTRGCT"):
    fold_dir = "%s_lr%s_batchsize%s" % (model_name, learning_rate, batch_size)
    params_path = os.path.join(exp_dir, fold_dir)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    best_val_loss = float("inf")
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        exp_fig_path = os.path.join(params_path, "figure")
        if not os.path.exists(exp_fig_path):
            os.makedirs(exp_fig_path)
        print("create experiment params directory %s" % params_path)
    else:
        print("experiment params directory exists")
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(epochs):
        net.train()
        train_loss_epoch_total = 0
        start_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            x, y = batch_data
            optimizer.zero_grad()  # 梯度清零
            outputs = net(s_w, x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5) # 梯度裁剪
            optimizer.step()
            train_loss_epoch_total += loss.item()
        train_loss = train_loss_epoch_total/len(train_loader) #一个epoch的训练损失
        val_prediction, val_loss = evaluate(net, s_w, val_loader)
        test_prediction, test_loss = evaluate(net, s_w, test_loader)
        # 训练集
        sw.add_scalar('training_loss', train_loss, epoch)
        # 验证集
        sw.add_scalar('valid_loss', val_loss.item(), epoch)
        # 测试集
        sw.add_scalar('test_loss', test_loss.item(), epoch)
        if val_loss < best_val_loss:
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            best_params_filename = os.path.join(params_path, 'best.params')
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), best_params_filename)
            if best_epoch//10 > 0 and best_epoch%10 >0:
                val_prediction = val_prediction.cpu().numpy()
                val_y = val_y_tensor.cpu().numpy()
                test_prediction = test_prediction.cpu().numpy()
                test_y = test_y_tensor.cpu().numpy()
                # if net.type == "auto":
                all_model_val_prediction_truth[model_name]['prediction'] = val_prediction.tolist()
                all_model_val_prediction_truth[model_name]['truth'] = val_y.tolist()
                all_model_test_prediction_truth[model_name]['prediction'] = test_prediction.tolist()
                all_model_test_prediction_truth[model_name]['truth'] = test_y.tolist()
                np.save(exp_dir + '/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
                np.save(exp_dir + '/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
                print('epoch %d,train_loss:%.8f,val_loss:%.8f, test_loss:%.8f, lr:%.6f, time:%.2f' % (best_epoch, train_loss, best_val_loss, test_loss,
                                                                                                      optimizer.state_dict()['param_groups'][0]['lr'], time.time() - start_time))
                scheduler.step()
            else:
                continue
        else:
            continue

def train_astgcn(net, train_loader, val_loader, test_loader,exp_dir, model_name="ASTGCN"):
    fold_dir = "%s_lr%s_batchsize%s" % (model_name, learning_rate, batch_size)
    params_path = os.path.join(exp_dir, fold_dir)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    best_val_loss = float("inf")
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        exp_fig_path = os.path.join(params_path, "figure")
        if not os.path.exists(exp_fig_path):
            os.makedirs(exp_fig_path)
        print("create experiment params directory %s" % params_path)
    else:
        print("experiment params directory exists")
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(epochs):
        net.train()
        train_loss_epoch_total = 0
        start_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            x, y = batch_data # astgcn的输入x是64, 307, 3,12,我们的是64,12,307,3,输出y:astgcn-->64,307,3(预测的步数），我们的是64,3,307,1
            x=x.permute(0,2,3,1) # (batch_size, num_of_vertices,in_channels, num_time_steps_input]
            y=y.squeeze(-1).permute(0,2,1) # [batch_size, num_of_vertices, num_time_steps_output]
            optimizer.zero_grad()  # 梯度清零
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5) # 梯度裁剪
            optimizer.step()
            train_loss_epoch_total += loss.item()
        train_loss = train_loss_epoch_total/len(train_loader) #一个epoch的训练损失
        val_prediction, val_loss = evaluate_astgcn(net, val_loader)
        test_prediction, test_loss = evaluate_astgcn(net, test_loader)
        # 训练集
        sw.add_scalar('training_loss', train_loss, epoch)
        # 验证集
        sw.add_scalar('valid_loss', val_loss.item(), epoch)
        # 测试集
        sw.add_scalar('test_loss', test_loss.item(), epoch)
        if val_loss < best_val_loss:
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            best_params_filename = os.path.join(params_path, 'best.params')
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), best_params_filename)
            if best_epoch//10 > 0 and best_epoch%10 >0:
                val_prediction = val_prediction.cpu().numpy()
                val_y = val_y_tensor.cpu().numpy()
                test_prediction = test_prediction.cpu().numpy()
                test_y = test_y_tensor.cpu().numpy()
                all_model_val_prediction_truth[model_name]['prediction'] = val_prediction.tolist()
                all_model_val_prediction_truth[model_name]['truth'] = val_y.tolist()
                all_model_test_prediction_truth[model_name]['prediction'] = test_prediction.tolist()
                all_model_test_prediction_truth[model_name]['truth'] = test_y.tolist()
                np.save(exp_dir + '/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
                np.save(exp_dir + '/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
                print('epoch %d,train_loss:%.8f,val_loss:%.8f, test_loss:%.8f, lr:%.6f, time:%.2f' % (best_epoch, train_loss, best_val_loss, test_loss,
                                                                                                      optimizer.state_dict()['param_groups'][0]['lr'], time.time() - start_time))
                scheduler.step()
            else:
                continue
        else:
            continue

def train_lstm(net, train_loader, val_loader, test_loader,exp_dir, model_name="LSTM"):
    fold_dir = "%s_lr%s_batchsize%s" % (model_name, learning_rate, batch_size)
    params_path = os.path.join(exp_dir, fold_dir)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    best_val_loss = float("inf")
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        exp_fig_path = os.path.join(params_path, "figure")
        if not os.path.exists(exp_fig_path):
            os.makedirs(exp_fig_path)
        print("create experiment params directory %s" % params_path)
    else:
        print("experiment params directory exists")
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(epochs):
        net.train()
        train_loss_epoch_total = 0
        start_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            x, y = batch_data # astgcn的输入x是64, 307, 3,12,我们的是64,12,307,3,输出y:astgcn-->64,307,3(预测的步数），我们的是64,3,307,1
            optimizer.zero_grad()  # 梯度清零
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5) # 梯度裁剪
            optimizer.step()
            train_loss_epoch_total += loss.item()
        train_loss = train_loss_epoch_total/len(train_loader) #一个epoch的训练损失
        val_prediction, val_loss = evaluate_lstm(net, val_loader)
        test_prediction, test_loss = evaluate_lstm(net, test_loader)
        # 训练集
        sw.add_scalar('training_loss', train_loss, epoch)
        # 验证集
        sw.add_scalar('valid_loss', val_loss.item(), epoch)
        # 测试集
        sw.add_scalar('test_loss', test_loss.item(), epoch)
        if val_loss < best_val_loss:
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            best_params_filename = os.path.join(params_path, 'best.params')
            best_val_loss = val_loss
            best_epoch = epoch+1
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), best_params_filename)
            if best_epoch//10 > 0 and best_epoch%10 >0:
                val_prediction = val_prediction.cpu().numpy()
                val_y = val_y_tensor.cpu().numpy()
                test_prediction = test_prediction.cpu().numpy()
                test_y = test_y_tensor.cpu().numpy()
                all_model_val_prediction_truth[model_name]['prediction'] = val_prediction.tolist()
                all_model_val_prediction_truth[model_name]['truth'] = val_y.tolist()
                all_model_test_prediction_truth[model_name]['prediction'] = test_prediction.tolist()
                all_model_test_prediction_truth[model_name]['truth'] = test_y.tolist()
                np.save(exp_dir + '/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
                np.save(exp_dir + '/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
                print('epoch %d,train_loss:%.8f,val_loss:%.8f, test_loss:%.8f, lr:%.6f, time:%.2f' % (best_epoch, train_loss, best_val_loss, test_loss,
                                                                                                      optimizer.state_dict()['param_groups'][0]['lr'], time.time() - start_time))
                scheduler.step()
            else:
                continue
        else:
            continue



if __name__ == "__main__":
    wp = WeightProcess(adj_file_name,num_of_vertices, type=dataset_name)
    A, A_distance, dist_mean, dist_std = wp.get_adjacency_matrix()
    s_w = wp.get_smooth_weight_matrix(A, A_distance, dist_mean, dist_std, scaling=True)
    s_w = torch.from_numpy(s_w).to(device)
    auto_gstrgct = GSTRGCT(num_of_vertices, in_channels, out_channels,
                           num_time_steps_input, num_time_steps_output,
                           dropout, K, d_model, n_heads, num_layers,
                           reg_mode="gsr", type='auto', tensor_deco=True).to(device)
    base_gstrgct = GSTRGCT(num_of_vertices,in_channels, out_channels,
                           num_time_steps_input, num_time_steps_output,
                           dropout, K, d_model=512, n_heads=8, num_layers=1,
                           reg_mode="gsr", type = 'base', tensor_deco=True).to(device)
    auto_strgct = GSTRGCT(num_of_vertices,in_channels, out_channels,
                          num_time_steps_input, num_time_steps_output,
                          dropout, K, d_model, n_heads, num_layers,
                          reg_mode="normal", type = 'auto', tensor_deco=True).to(device)
    auto_gstrgct_cas = GSTRGCT(num_of_vertices,in_channels, out_channels,
                               num_time_steps_input, num_time_steps_output,
                               dropout, K, d_model, n_heads, num_layers,
                               reg_mode="gsr", type = 'auto', tensor_deco=False).to(device)
    astgcn = make_model(device, nb_block=2, in_channels=3, K=1, nb_chev_filter=64, nb_time_filter=64,
                        time_strides=1, adj_mx=s_w.cpu().numpy(), num_for_predict=3, len_input=12, num_of_vertices=num_of_vertices)
    stgcn = STGCN(num_nodes=num_of_vertices, num_features=3, num_timesteps_input=12, num_timesteps_output=3).to(device)
    # lstm = LSTM(num_time_steps_in=12, num_time_steps_out=3, in_channels=3, hidden_size=64, out_channels=1, num_layers=2).to(device)
    nets = [auto_gstrgct, base_gstrgct, auto_strgct, auto_gstrgct_cas, stgcn]
    # nets = [base_gstrgct, auto_gstrgct_cas]

    model_names = ["Auto-GSTRGCT","Base-GSTRGCT", "Auto-STRGCT", "Auto-GSTRGCT_Cas", "STGCN"]
    # for net, model_name in zip(nets, model_names):
    #     train(net, s_w, train_loader, val_loader, test_loader, exp_dir, model_name=model_name)
    # train(stgcn, train_loader, val_loader, test_loader, exp_dir, model_name='STGCN')
    train_astgcn(astgcn, train_loader, val_loader, test_loader,exp_dir, model_name="ASTGCN")
    # train(stgcn, s_w, train_loader, val_loader, test_loader, exp_dir, model_name="STGCN")






