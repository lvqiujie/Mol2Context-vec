import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tasks.FreeSolv.train import LSTM, MyDataset
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
import numpy as np

# 设置超参数
input_size = 512
num_layers = 2                                  #定义超参数rnn的层数，层数为1层
hidden_size = 512                                #定义超参数rnn的循环神经元个数，个数为32个
learning_rate = 0.02                            #定义超参数学习率
epoch_num = 1000
batch_size = 1
best_loss = 10000
test_best_loss = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# filepath="FreeSolv/delaney.csv"
# df = pd.read_csv(filepath, header=0, encoding="gbk")

y = joblib.load('FreeSolv/label.pkl')
all_smi = np.array(joblib.load('FreeSolv/smi.pkl'))
x = joblib.load('FreeSolv/FreeSolv_embed.pkl')

data_test = MyDataset(x, y,all_smi)
dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True, drop_last=True)
rnn = LSTM().to(device)
rnn_dict = torch.load('FreeSolv/lstm_net.pth')
task_matrix = torch.load('FreeSolv/task_matrix.pth')
# print(task_matrix[0], task_matrix[1], task_matrix[2])
rnn.load_state_dict(rnn_dict)
rnn.cuda()
rnn.eval()
for tmp_x, tmp_y, tmp_smi in dataset_test:
    out, alpha_n, att_n = rnn(tmp_x, task_matrix)
    print(tmp_smi[0],out.cpu().detach().numpy()[0], tmp_y.cpu().detach().numpy()[0], )