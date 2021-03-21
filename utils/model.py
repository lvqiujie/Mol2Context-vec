import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """搭建rnn网络"""
    def __init__(self, out_num, input_size=300, task_type='sing', att=False):
        super(LSTM, self).__init__()
        self.matrix = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        self.input_size = input_size
        self.out_num = out_num * 2 if "muti".__eq__(task_type) else out_num
        self.att = att

        self.fc = nn.Linear(self.input_size, 1024)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,)
            # bidirectional=True)
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, self.out_num)
        self.dropout = nn.Dropout(p=0.3)
        # self.sig = nn.Sigmoid()
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(128)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)

        return context, alpha_n

    def forward(self, x):
        # bs = len(x)
        # length = np.array([t.shape[0] for t in x])
        #
        # x, orderD = pack_sequences(x)
        # print(self.matrix[0],self.matrix[1],self.matrix[2])
        x = x.to(device)
        x = self.matrix[0] * x[:, 0, :, :] + self.matrix[1] * x[:, 1, :, :] + self.matrix[2] * x[:, 2, :, :]
        x = self.fc(x.to(device)).to(device)
        # changed_length1 = length[orderD]
        # x = pack_padded_sequence(x, changed_length1, batch_first=True)

        out,(h_n, c_n) = self.lstm(x.to(device))     #h_state是之前的隐层状态
        # out = torch.cat((h_n[-1, :, :], h_n[-2, :, :]), dim=-1)
        # out1 = unpack_sequences(rnn_out, orderD)
        # for i in range(bs):
        #     out1[i,length[i]:-1,:] = 0

        if self.att:
            query = self.dropout(out)

            # 加入attention机制
            out, alpha_n = self.attention_net(out, query)

        else:
            out = torch.mean(out,dim=1).squeeze().cuda()
            # out = out[:,-1,:]


        #进行全连接
        # out = self.fc1(out[:,-1,:])
        # out = F.relu(out)
        # out = self.bn1(F.dropout(out, p=0.3))
        # out = self.fc2(out)
        # out = F.relu(out)
        # out = self.bn2(F.dropout(out, p=0.3))
        out = self.fc3(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        # return F.softmax(out,dim=-1)
        return out

class MyDataset(data.Dataset):
    def __init__(self, compound, y, smi):
        super(MyDataset, self).__init__()
        self.compound = compound
        # self.compound = torch.FloatTensor(compound)
        # self.y = torch.FloatTensor(y)
        self.y = y
        self.smi = smi

    def __getitem__(self, item):
        return self.compound[item], self.y[item], self.smi[item]


    def __len__(self):
        return len(self.compound)