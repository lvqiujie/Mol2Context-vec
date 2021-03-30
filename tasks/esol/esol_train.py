from rdkit import Chem
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import seaborn as sns
import math
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_len(smi):
    mol = Chem.MolFromSmiles(smi)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    return len(mol_atoms)

def pack_sequences(X, order=None):
    lengths = np.array([x.shape[0] for x in X])
    features = X[0].shape[1]
    n = len(X)
    if order is None:
        order = np.argsort(lengths)[::-1]  # 从后向前取反向的元素
    m = max(lengths)

    X_block = X[0].new(n, m, features).zero_()

    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i, :len(x), :] = x

    return X_block, order


def unpack_sequences(X, order):
    X, lengths = pad_packed_sequence(X, batch_first=True)
    X_block = torch.zeros(size=X.size()).to(device)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i]
    return X_block

def split_data(x, y, all_smi, lens, k_fold):
    y = np.array(y, dtype=np.float64)
    all_smi = np.array(all_smi)
    lens = np.array(lens)

    # save_path = 'esol/'+str(k_fold)+'-fold-index.pkl'
    # if os.path.isfile(save_path):
    #     index = joblib.load(save_path)
    #     train_split_x = x[index["train_index"]]
    #     train_split_y = y[index["train_index"]]
    #     val_split_x = x[index["val_index"]]
    #     val_split_y = y[index["val_index"]]
    #     test_split_x = x[index["test_index"]]
    #     test_split_y = y[index["test_index"]]
    #     train_weights = joblib.load('esol/train_weights.pkl')
    #     return train_split_x, train_split_y, val_split_x, val_split_y, test_split_x, test_split_y, train_weights

    kf = KFold(4, True, 100)
    train_index = [[],[],[],[],[]]
    val_index = [[],[],[],[],[]]
    test_index = [[],[],[],[],[]]
    for k, tmp in enumerate(kf.split(x)):
        # train_tmp is  the index ofnegative_index
        train_tmp, test_tmp = tmp
        train_index[k].extend(train_tmp)
        num_t = int(len(test_tmp)/2)
        val_index[k].extend(test_tmp[0:num_t])
        test_index[k].extend(test_tmp[num_t:])


    for i in range(5):
        joblib.dump({"train_index":train_index[i],
                     "val_index": val_index[i],
                     "test_index": test_index[i],
                     }, 'esol/'+str(i+1)+'-fold-index.pkl')
    train_split_x = x[train_index[k_fold]]
    train_split_y = y[train_index[k_fold]]
    train_split_smi = all_smi[train_index[k_fold]]
    train_split_lens = lens[train_index[k_fold]]
    val_split_x = x[val_index[k_fold]]
    val_split_y = y[val_index[k_fold]]
    val_split_smi = all_smi[val_index[k_fold]]
    val_split_lens = lens[val_index[k_fold]]
    test_split_x = x[test_index[k_fold]]
    test_split_y = y[test_index[k_fold]]
    test_split_smi = all_smi[test_index[k_fold]]
    test_split_lens = lens[test_index[k_fold]]
    return train_split_x, train_split_y, train_split_smi, train_split_lens,\
           val_split_x, val_split_y, val_split_smi,val_split_lens,\
           test_split_x, test_split_y, test_split_smi,test_split_lens

class LSTM(nn.Module):
    """搭建rnn网络"""
    def __init__(self):
        super(LSTM, self).__init__()
        self.matrix = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        self.fc = nn.Linear(600, 1024)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,)

        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=0.5)



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

        att = torch.matmul(x, context.unsqueeze(2))/ math.sqrt(d_k)
        att = torch.sigmoid(att.squeeze())
        return context, alpha_n, att


    def forward(self, x, x_lens):
        # print(self.matrix1, self.matrix2, self.matrix3)
        # bs = len(x)
        # length = np.array([t.shape[0] for t in x])
        #
        # x, orderD = pack_sequences(x)
        # x = self.matrix1 * x[:,0,:,:] + self.matrix2 * x[:,1,:,:] + self.matrix3 * x[:,2,:,:]
        x = x.to(device)
        x = self.matrix[0] * x[:, 0, :, :] + self.matrix[1] * x[:, 1, :, :] + self.matrix[2] * x[:, 2, :, :]

        x = self.fc(x.to(device)).to(device)
        # packing
        # embed_packed = pack_padded_sequence(x, x_lens,
        #                                     batch_first=True,
        #                                     enforce_sorted=False)

        out, (hidden, cell) = self.lstm(x)     #h_state是之前的隐层状态

        query = self.dropout(out)

        # 加入attention机制
        out_att, alpha_n, att = self.attention_net(out, query)

        # alpha_n =0
        # att =0
        # out,hidden = self.lstm(x.to(device))     #h_state是之前的隐层状态
        # out = torch.cat((h_n[-1, :, :], h_n[-2, :, :]), dim=-1)
        # out1 = unpack_sequences(rnn_out, orderD)
        # for i in range(bs):
        #     out1[i,length[i]:-1,:] = 0
        out = torch.mean(out, dim=1).squeeze()
        # out = out[:,-1,:]


        #进行全连接
        out_tmp = self.fc3(out)
        out_tmp = F.leaky_relu(out_tmp)
        out_tmp = self.dropout(out_tmp)
        out = self.fc4(out_tmp)

        # outputs = []
        # for i, out_tmp in enumerate(out):
        #     # out_tmp = torch.mean(out_tmp[:lens[i],:], dim=0).squeeze()
        #     out_tmp = out_tmp[lens[i]-1,:]
        #     out_tmp = self.fc3(out_tmp)
        #     out_tmp = F.leaky_relu(out_tmp)
        #     out_tmp = self.dropout(out_tmp)
        #     out_tmp = self.fc4(out_tmp)
        #     outputs.append(out_tmp)
        # out = torch.stack(outputs, dim=0)
        return out, alpha_n, att

class MyDataset(data.Dataset):
    def __init__(self, compound, y, smi, len):
        super(MyDataset, self).__init__()
        self.compound = compound
        # self.compound = torch.FloatTensor(compound)
        # self.y = torch.FloatTensor(y)
        self.y = y
        self.smi = smi
        self.len = len

    def __getitem__(self, item):
        return self.compound[item], self.y[item], self.smi[item], self.len[item]


    def __len__(self):
        return len(self.compound)

if __name__ == '__main__':
    # best 0 0.01 16 188  100
    # 设置超参数
    input_size = 512
    num_layers = 2  # 定义超参数rnn的层数，层数为1层
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.01  # 定义超参数学习率
    epoch_num = 1000
    batch_size = 16
    best_loss = 10000
    test_best_loss = 1000
    weight_decay = 1e-5
    momentum = 0.9

    b = 0.051
    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # filepath = "esol/delaney.csv"
    # df = pd.read_csv(filepath, header=0, encoding="gbk")
    y = joblib.load('esol/label.pkl')
    all_smi = np.array(joblib.load('esol/smi.pkl'))

    x = joblib.load('esol/esol_embed.pkl')
    lens = joblib.load('esol/lens.pkl')

    # 5-Fold
    train_split_x, train_split_y, train_split_smi, train_split_lens,\
    val_split_x, val_split_y, val_split_smi, val_split_lens,\
    test_split_x, test_split_y, test_split_smi, test_split_lens = split_data(x, y, all_smi, lens, 0)

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi, train_split_lens)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi, val_split_lens)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi, test_split_lens)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)


    rnn = LSTM().to(device)


    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    # optimizer = torch.optim.Adam(list(rnn.parameters()), lr=learning_rate)
    optimizer = torch.optim.SGD(list(rnn.parameters()),
                                lr=learning_rate, weight_decay = weight_decay,
                                momentum = momentum)
    loss_function = nn.MSELoss().to(device)

    # 按照以下的过程进行参数的训练
    for epoch in range(epoch_num):
        avg_loss = 0
        sum_loss = 0
        rnn.train()
        # print(task_matrix[0], task_matrix[1], task_matrix[2])
        for index, tmp in enumerate(dataset_train):
            tmp_compound, tmp_y, tmp_smi, tmp_len = tmp
            optimizer.zero_grad()
            outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device))
            # print(matrix1,matrix2,matrix3)
            # print(outputs.flatten())
            loss = loss_function(outputs.flatten(), tmp_y.type(torch.FloatTensor).to(device))

            # loss = (loss - b).abs() + b
            loss.backward()
            optimizer.step()

            sum_loss += loss
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        avg_loss = sum_loss / (index + 1)
        print("epoch:", epoch,"   train  "  "avg_loss:", avg_loss.item())
        # # 保存模型
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     PATH = 'esol/lstm_net.pth'
        #     print("train save model")
        #     torch.save(rnn.state_dict(), PATH)

        # print(task_matrix[0], task_matrix[1], task_matrix[2])
        with torch.no_grad():
            rnn.eval()
            test_avg_loss = 0
            test_sum_loss = 0
            for index, tmp in enumerate(dataset_val):
                tmp_compound, tmp_y, tmp_smi, tmp_len = tmp

                outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device))
                # print(outputs.flatten())
                loss = loss_function(outputs.flatten(), tmp_y.type(torch.FloatTensor).to(device))
                test_sum_loss += loss.item()


            test_avg_loss = test_sum_loss / (index + 1)
            print("epoch:", epoch,"   val  ", "avg_loss: ", test_avg_loss)
            # 保存模型
            if test_avg_loss < test_best_loss:
                test_best_loss = test_avg_loss
                print("test save model")
                torch.save(rnn.state_dict(), 'esol/lstm_net.pth')
                att_flag = False
                # if test_avg_loss < 0.5:
                #     att_flag = True
                # print(task_matrix[0], task_matrix[1], task_matrix[2])
            rnn.eval()
            test_avg_loss = 0
            test_sum_loss = 0
            all_pred = []
            all_label = []

            for index, tmp in enumerate(dataset_test):
                tmp_compound, tmp_y, tmp_smi, tmp_len = tmp
                loss = 0
                outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device))


                y_pred = outputs.to(device).view(-1)
                y_label = tmp_y.float().to(device).view(-1)

                all_label.extend(y_label.cpu().numpy())
                all_pred.extend(y_pred.cpu().numpy())

                # y_pred = torch.sigmoid(y_pred).view(-1)
                # y_label = F.one_hot(y_label, 2).float().to(device)
                loss += loss_function(y_pred, y_label)

                test_sum_loss += loss.item()

            mae = mean_absolute_error(all_label, all_pred)
            mse = mean_squared_error(all_label, all_pred)
            rmse = np.sqrt(mse)
            test_avg_loss = test_sum_loss / (index + 1)

            print("epoch:", epoch, "   test   avg_loss:", test_avg_loss
                  ," mae : ", mae
                  ," rmse : ", rmse)


