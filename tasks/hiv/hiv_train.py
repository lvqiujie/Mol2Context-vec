import sys
sys.path.append('./')
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import math
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# from utils.util import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_data(x, y, all_smi, k_fold, name):
    y = np.array(y)
    all_smi = np.array(all_smi)
    # save_path = 'hiv/'+str(k_fold)+'-fold-index.pkl'
    # if os.path.isfile(save_path):
    #     index = joblib.load(save_path)
    #     train_split_x = x[index["train_index"]]
    #     train_split_y = y[index["train_index"]]
    #     val_split_x = x[index["val_index"]]
    #     val_split_y = y[index["val_index"]]
    #     test_split_x = x[index["test_index"]]
    #     test_split_y = y[index["test_index"]]
    #     train_weights = joblib.load('hiv/train_weights.pkl')
    #     return train_split_x, train_split_y, val_split_x, val_split_y, test_split_x, test_split_y, train_weights

    kf = KFold(5, True, 100)
    train_index = [[],[],[],[],[]]
    val_index = [[],[],[],[],[]]
    test_index = [[],[],[],[],[]]
    negative_index = np.where(y == 0)[0]
    positive_index = np.where(y == 1)[0]
    for k, tmp in enumerate(kf.split(negative_index)):
        # train_tmp is  the index ofnegative_index
        train_tmp, test_tmp = tmp
        train_index[k].extend(negative_index[train_tmp])
        num_t = int(len(test_tmp) / 2)
        val_index[k].extend(negative_index[test_tmp[:num_t]])
        test_index[k].extend(negative_index[test_tmp[num_t:]])
    for k, tmp in enumerate(kf.split(positive_index)):
        train_tmp, test_tmp = tmp
        train_index[k].extend(positive_index[train_tmp])
        num_t = int(len(test_tmp) / 2)
        val_index[k].extend(positive_index[test_tmp[:num_t]])
        test_index[k].extend(positive_index[test_tmp[num_t:]])
    weights = [(len(negative_index) + len(positive_index)) / len(negative_index),
                                           (len(negative_index) + len(positive_index)) / len(positive_index)]

    for i in range(5):
        joblib.dump({"train_index":train_index[i],
                     "val_index": val_index[i],
                     "test_index": test_index[i],
                     }, name+'/'+str(i+1)+'-fold-index.pkl')
    joblib.dump(weights, name + '/weights.pkl')
    train_split_x = x[train_index[k_fold]]
    train_split_y = y[train_index[k_fold]]
    train_split_smi = all_smi[train_index[k_fold]]
    val_split_x = x[val_index[k_fold]]
    val_split_y = y[val_index[k_fold]]
    val_split_smi = all_smi[val_index[k_fold]]
    test_split_x = x[test_index[k_fold]]
    test_split_y = y[test_index[k_fold]]
    test_split_smi = all_smi[test_index[k_fold]]
    return train_split_x, train_split_y, train_split_smi,\
           val_split_x, val_split_y, val_split_smi,\
           test_split_x, test_split_y, test_split_smi, weights


# binary class
class CELoss(nn.Module):

    def __init__(self, weight=2):
        super(CELoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        target = target.float()
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        # print(loss)
        return loss.mean()

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        target = target.float()
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        # print(loss)
        return loss.mean()

class LSTM(nn.Module):
    """搭建rnn网络"""
    def __init__(self):
        super(LSTM, self).__init__()
        self.matrix = nn.Parameter(torch.tensor([0.33,0.33,0.33]), requires_grad=True)
        self.fc = nn.Linear(300, 1024)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,)
            # bidirectional=True)
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
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
        x = self.matrix[0] * x[:, 0, :, :] + self.matrix[1] * x[:, 1, :, :] + self.matrix[2] * x[:, 2, :, :]
        x = self.fc(x.to(device)).to(device)
        # changed_length1 = length[orderD]
        # x = pack_padded_sequence(x, changed_length1, batch_first=True)

        out,(h_n, c_n) = self.lstm(x.to(device))     #h_state是之前的隐层状态
        # out = torch.cat((h_n[-1, :, :], h_n[-2, :, :]), dim=-1)
        # out1 = unpack_sequences(rnn_out, orderD)
        # for i in range(bs):
        #     out1[i,length[i]:-1,:] = 0
        out = torch.mean(out, dim=1).squeeze().cuda()
        # out = out[:,-1,:]

        # query = self.dropout(out)
        #
        # # 加入attention机制
        # attn_output, alpha_n = self.attention_net(out, query)

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

if __name__ == '__main__':
    # 设置超参数
    input_size = 512
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.01  # 定义超参数学习率
    epoch_num = 2000
    batch_size = 512
    best_loss = 10000
    test_best_loss = 10000
    weight_decay = 1e-5
    momentum = 0.9

    b = 0.3
    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    y = joblib.load('hiv/label.pkl')

    x = joblib.load('hiv/hiv_embed.pkl')
    all_smi = joblib.load('hiv/smi.pkl')
    print("data len is ",x.shape[0])

    # 5-Fold
    train_split_x, train_split_y, train_split_smi, \
    val_split_x, val_split_y, val_split_smi, \
    test_split_x, test_split_y, test_split_smi, weights = split_data(x, y, all_smi, 3, "hiv")

    # print(weights)

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True,drop_last=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    rnn = LSTM().to(device)
    # 设置优化器和损失函数
    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay,
    #                             momentum = momentum)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay)
    # loss_function = F.cross_entropy
    # loss_function = F.nll_loss
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights[1]).to(device)).to(device)
    # loss_function = [FocalLoss(alpha=1 / w[0]) for w in train_weights]
    # loss_function = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(device), reduction='mean')
    #                  for w in train_weights]

    # 按照以下的过程进行参数的训练
    for epoch in range(epoch_num):
        avg_loss = 0
        sum_loss = 0
        rnn.train()
        y_true = []
        y_pred = []
        y_pred_score = []

        for index, tmp in enumerate(dataset_train):
            tmp_compound, tmp_y, tmp_smi = tmp
            optimizer.zero_grad()
            outputs = rnn(tmp_compound.to(device))



            outputs = outputs.to(device).view(-1)
            y_label = tmp_y.float().to(device).view(-1)


            # print(outputs.shape, tmp_y.shape)
            loss = loss_function(outputs.to(device), y_label.float().to(device))

            outputs_score = torch.sigmoid(outputs).view(-1)
            pred = np.zeros_like(outputs.cpu().detach().numpy(), dtype=int)
            pred[np.where(np.asarray(outputs_score.cpu().detach().numpy()) > 0.5)] = 1

            y_pred.extend(pred)
            y_true.extend(y_label.cpu().numpy())
            y_pred_score.extend(outputs_score.cpu().detach().numpy())
            # flood = (loss - b).abs() + b
            loss.backward()
            optimizer.step()

            sum_loss += loss
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        avg_loss = sum_loss / (index + 1)
        cm = metrics.confusion_matrix(y_true, y_pred)
        print("epoch:", epoch,"   train  "  "avg_loss:", avg_loss.item(),
                # "acc: ", metrics.accuracy_score(y_true, y_pred),
                # "recall: ", metrics.recall_score(y_true, y_pred),
                # "specificity: ", cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                " train_auc: ", metrics.roc_auc_score(y_true, y_pred_score))

        # 测试集部分换机器后再调试
        with torch.no_grad():
            rnn.eval()
            val_avg_loss = 0
            val_sum_loss = 0
            y_true = []
            y_pred = []
            y_pred_score = []
            for index, tmp in enumerate(dataset_val):
                tmp_compound, tmp_y, tmp_sm = tmp

                y_true.extend(tmp_y.cpu().numpy())
                outputs = rnn(tmp_compound)
                # out_label = F.softmax(outputs, dim=1)
                # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
                # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i, x in enumerate(out_label.cpu().detach().numpy())]
                # y_pred.extend(pred)
                # y_pred_score.extend(pred_score)

                outputs = outputs.to(device).view(-1)
                y_label = tmp_y.float().to(device).view(-1)
                loss = loss_function(outputs, y_label)


                pred_score = torch.sigmoid(outputs)
                pred = np.zeros_like(outputs.cpu().detach().numpy(), dtype=int)
                pred[np.where(np.asarray(pred_score.cpu().detach().numpy()) > 0.5)] = 1
                y_pred.extend(pred)
                y_pred_score.extend(pred_score.cpu().detach().numpy())

                val_sum_loss += loss.item()


            val_avg_loss = val_sum_loss / (index + 1)
            cm = metrics.confusion_matrix(y_true, y_pred)
            print("epoch:", epoch,"   val  ", "avg_loss: ", val_avg_loss,
                  "acc: ", metrics.accuracy_score(y_true, y_pred),
                  # "recall: ", metrics.recall_score(y_true, y_pred),
                  # "specificity: ", cm[0,0]/(cm[0,0]+cm[0,1]),
                  # "sensitivity: ", cm[1,1]/(cm[1,0]+cm[1,1]),
                  " val_auc: ", metrics.roc_auc_score(y_true, y_pred_score))
            # 保存模型
            if val_avg_loss < test_best_loss:
                test_best_loss = val_avg_loss
                PATH = 'hiv/lstm_net.pth'
                print("test save model")
                torch.save(rnn.state_dict(), PATH)

                with torch.no_grad():
                    rnn.eval()
                    test_avg_loss = 0
                    test_sum_loss = 0
                    y_true_task = []
                    y_pred_task = []
                    y_pred_task_score = []
                    for index, tmp in enumerate(dataset_test):
                        tmp_compound, tmp_y, tmp_smi = tmp
                        loss = 0
                        outputs = rnn(tmp_compound)
                        # out_label = F.softmax(outputs, dim=1)
                        # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
                        # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i, x in enumerate(out_label.cpu().detach().numpy())]
                        # y_pred.extend(pred)
                        # y_pred_score.extend(pred_score)

                        y_pred = outputs.to(device).view(-1)
                        y_label = tmp_y.float().to(device).view(-1)

                        # y_pred = torch.sigmoid(y_pred).view(-1)
                        # y_label = F.one_hot(y_label, 2).float().to(device)
                        loss += loss_function(y_pred, y_label)

                        # pred_score = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

                        pred_score = torch.sigmoid(y_pred.detach().cpu()).view(-1).numpy()

                        pred_lable = np.zeros_like(pred_score, dtype=int)
                        pred_lable[np.where(np.asarray(pred_score) > 0.5)] = 1

                        y_true_task.extend(y_label.cpu().numpy())
                        y_pred_task.extend(pred_lable)
                        y_pred_task_score.extend(pred_score)

                        test_sum_loss += loss.item()

                    test_avg_loss = test_sum_loss / (index + 1)
                    cm = metrics.confusion_matrix(y_true_task, y_pred_task)
                    trn_roc = metrics.roc_auc_score(y_true_task, y_pred_task_score)
                    trn_prc = metrics.auc(precision_recall_curve(y_true_task, y_pred_task_score)[1],
                                          precision_recall_curve(y_true_task, y_pred_task_score)[0])
                    acc = metrics.accuracy_score(y_true_task, y_pred_task)
                    recall = metrics.recall_score(y_true_task, y_pred_task)
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

                    print("epoch:", epoch, "   test   avg_loss:", test_avg_loss,
                          "acc: ", np.array(acc).mean(),
                          # "recall: ", np.array(recall).mean(),
                          # "specificity: ", np.array(specificity).mean(),
                          " test_auc: ", np.array(trn_roc).mean(),
                          " test_pr: ", np.array(trn_prc).mean())
