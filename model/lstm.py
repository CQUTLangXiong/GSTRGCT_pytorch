import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    """
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, num_time_steps_in=12, num_time_steps_out=3, in_channels=3, hidden_size=64, out_channels=1, num_layers=2):
        super().__init__()
        self.num_time_steps_in = num_time_steps_in
        self.num_time_steps_out = num_time_steps_out
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.lstm = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.out_channels)
        self.t_linear = nn.Linear(self.num_time_steps_in, self.num_time_steps_out)

    def forward(self, x): # x: 64,12,307,3
        x = x.permute(2,1,0,3) # 307, 12, 64,3-->[num_of_vertices, num_time_steps_in, batch_size, in_channels]
        out =[]
        for i in range(x.shape[0]): # 对每个节点预测 无空间依赖
            v_x = x[i, :, :, :] # [num_time_steps_in, batch_size, in_channels]
            v_out, _ = self.lstm(v_x) # [num_time_steps_in, batch_size, hidden_size]
            v_out = F.relu(self.fc(v_out)) # [num_time_steps_in, batch_size, out_channels]
            t_out = self.t_linear(v_out.permute(1, 2, 0)) # [batch_size, out_channels,num_time_steps_out]
            out.append(t_out.unsqueeze(0))
        out = torch.cat(out, dim=0) # [num_of_vertices,batch_size, out_channels,num_time_steps_out]
        out= out.permute(1,3,0,2)# [batch_size,num_time_steps_out,num_of_vertices, out_channels]
        return out

if __name__ == '__main__':
    x = torch.rand(64, 12,307, 3)
    lstm = LSTM(num_time_steps_in=12, num_time_steps_out=3, in_channels=3, hidden_size=64, out_channels=1, num_layers=2)
    out = lstm(x)
    print(out.shape)
