import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve
import numpy as np
import math
import random
from sklearn import metrics
# from utils.util import *
# from utils.model import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
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

def split_multi_label(x, y, smi, k_fold, name):
    y = np.array(y).astype(float)
    all_smi = np.array(smi)
    # save_path = 'tox/'+str(k_fold)+'-fold-index.pkl'
    # if os.path.isfile(save_path):
    #     index = joblib.load(save_path)
    #     train_split_x = x[index["train_index"]]
    #     train_split_y = y[index["train_index"]]
    #     val_split_x = x[index["val_index"]]
    #     val_split_y = y[index["val_index"]]
    #     test_split_x = x[index["test_index"]]
    #     test_split_y = y[index["test_index"]]
    #     train_weights = joblib.load('tox/train_weights.pkl')
    #     return train_split_x, train_split_y, val_split_x, val_split_y, test_split_x, test_split_y, train_weights

    kf = KFold(5, False, 100)
    all_train_index = [[],[],[],[],[]]
    all_train_index_weights = [[] for i in range(y.shape[1])]
    all_val_index = [[],[],[],[],[]]
    all_test_index = [[],[],[],[],[]]
    for task_index in range(y.shape[-1]):
        negative_index = np.where(y[:, task_index] == 0)[0]
        positive_index = np.where(y[:, task_index] == 1)[0]
        train_index = [[],[],[],[],[]]
        val_index = [[],[],[],[],[]]
        test_index = [[],[],[],[],[]]
        for k, tmp in enumerate(kf.split(negative_index)):
            # train_tmp is  the index ofnegative_index
            train_tmp, test_tmp = tmp
            train_index[k].extend(negative_index[train_tmp])
            num_t = int(len(test_tmp)/2)
            val_index[k].extend(negative_index[test_tmp[:num_t]])
            test_index[k].extend(negative_index[test_tmp[num_t:]])
        for k, tmp in enumerate(kf.split(positive_index)):
            train_tmp, test_tmp = tmp
            train_index[k].extend(positive_index[train_tmp])
            num_t = int(len(test_tmp)/2)
            val_index[k].extend(positive_index[test_tmp[:num_t]])
            test_index[k].extend(positive_index[test_tmp[num_t:]])

        all_train_index_weights[task_index] = [(len(negative_index) + len(positive_index)) / len(negative_index),
                                               (len(negative_index) + len(positive_index)) / len(positive_index)]

        if task_index == 0:
            all_train_index = train_index
            all_val_index = val_index
            all_test_index = test_index
        else:
            all_train_index = [list(set(all_train_index[i]).union(set(t))) for i, t in enumerate(train_index)]
            all_val_index = [list(set(all_val_index[i]).union(set(t))) for i, t in enumerate(val_index)]
            all_test_index = [list(set(all_test_index[i]).union(set(t))) for i, t in enumerate(test_index)]
    for i in range(5):
        joblib.dump({"train_index":all_train_index[i],
                     "val_index": all_val_index[i],
                     "test_index": all_test_index[i],
                     }, name+'/'+str(i+1)+'-fold-index.pkl')
    joblib.dump(all_train_index_weights, name+'/weights.pkl')
    train_split_x = x[all_train_index[k_fold]]
    train_split_y = y[all_train_index[k_fold]]
    train_split_smi = all_smi[all_train_index[k_fold]]
    val_split_x = x[all_val_index[k_fold]]
    val_split_y = y[all_val_index[k_fold]]
    val_split_smi = all_smi[all_val_index[k_fold]]
    test_split_x = x[all_test_index[k_fold]]
    test_split_y = y[all_test_index[k_fold]]
    test_split_smi = all_smi[all_test_index[k_fold]]
    return train_split_x, train_split_y, train_split_smi,\
           val_split_x, val_split_y, val_split_smi,\
           test_split_x, test_split_y, test_split_smi, all_train_index_weights


if __name__ == '__main__':
    # 设置超参数
    input_size = 512
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.01  # 定义超参数学习率
    epoch_num = 2000
    batch_size = 128
    best_loss = 10000
    test_best_loss = 10000
    weight_decay = 1e-5
    momentum = 0.9

    b = 0.2
    dict_label = {"NR-AR": 0,
                  "NR-AR-LBD": 1,
                  "NR-AhR": 2,
                  "NR-Aromatase": 3,
                  "NR-ER": 4,
                  "NR-ER-LBD": 5,
                  "NR-PPAR-gamma": 6,
                  "SR-ARE": 7,
                  "SR-ATAD5": 8,
                  "SR-HSE": 9,
                  "SR-MMP": 10,
                  "SR-p53": 11, }
    tasks = list(dict_label.keys())
    tasks_num = len(tasks)

    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    y = joblib.load("tox/label.pkl")
    y = np.array(y).astype(float)
    print(y.shape)
    all_smi = joblib.load("tox/smi.pkl")

    x = joblib.load("tox/tox_embed.pkl")

    # 5-Fold
    train_split_x, train_split_y, train_split_smi, \
    val_split_x, val_split_y, val_split_smi, \
    test_split_x, test_split_y, test_split_smi, weights = split_multi_label(x, y, all_smi, 3, 'tox')

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    rnn = LSTM(tasks_num, task_type="muti", input_size=300).to(device)
    # 设置优化器和损失函数
    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                momentum=momentum)
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.Adadelta(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay, rho=0.9)
    # optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # loss_function = F.cross_entropy
    # loss_function = F.nll_loss
    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight).to(device), reduction='mean') for weight in weights]
    # loss_function = nn.BCELoss()
    # loss_function = nn.BCEWithLogitsLoss()

    # 按照以下的过程进行参数的训练
    for epoch in range(epoch_num):
        avg_loss = 0
        sum_loss = 0
        rnn.train()
        y_true_task = {}
        y_pred_task = {}
        y_pred_task_score = {}
        for index, tmp in enumerate(dataset_train):
            tmp_compound, tmp_y, tmp_smi = tmp
            # tmp_y = tmp_y.float()
            optimizer.zero_grad()
            outputs = rnn(tmp_compound.to(device))
            loss = 0
            for i in range(len(tasks)):
                validId = np.where((tmp_y[:, i].cpu().numpy() == 0) | (tmp_y[:, i].cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                y_pred = outputs[:, i * 2:(i + 1) * 2][torch.tensor(validId).to(device)]
                y_label = tmp_y[:, i][torch.tensor(validId).to(device)]

                # y_pred = torch.sigmoid(y_pred).view(-1)
                # y_label = F.one_hot(y_label, 2).float().to(device)
                loss += loss_function[i](y_pred.to(device), y_label.long().to(device))

                pred_lable = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                # pred_lable = np.zeros_like(y_pred.cpu().detach().numpy(), dtype=int)
                # pred_lable[np.where(np.asarray(y_pred.cpu().detach().numpy()) > 0.5)] = 1
                try:
                    y_true_task[i].extend(y_label.cpu().numpy())
                    y_pred_task[i].extend(pred_lable)
                    # y_pred_task_score[i].extend(y_pred)
                except:
                    y_true_task[i] = []
                    y_pred_task[i] = []
                    # y_pred_task_score[i] = []
                    y_true_task[i].extend(y_label.cpu().numpy())
                    y_pred_task[i].extend(pred_lable)
                    # y_pred_task_score[i].extend(y_pred.cpu().detach().numpy())

                # flood = (loss - b).abs() + b

            loss.backward()
            optimizer.step()

            sum_loss += loss
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        avg_loss = sum_loss / (index + 1)
        # cm = [metrics.confusion_matrix(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        trn_roc = [metrics.roc_auc_score(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        trn_prc = [metrics.auc(precision_recall_curve(y_true_task[i], y_pred_task[i])[1],
                               precision_recall_curve(y_true_task[i], y_pred_task[i])[0]) for i in range(len(tasks))]
        # acc = [metrics.accuracy_score(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        # recall = [metrics.recall_score(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        # specificity = [cm[i][0, 0] / (cm[i][0, 0] + cm[i][0, 1]) for i in range(len(tasks))]

        print("epoch:", epoch, "   train  "  "avg_loss:", avg_loss.item(),
              # "acc: ", np.array(acc).mean(),
              # "recall: ", np.array(recall).mean(),
              # "specificity: ", np.array(specificity).mean(),
              " train_auc: ", np.array(trn_roc).mean(),
              " train_pr: ", np.array(trn_prc).mean())

        with torch.no_grad():
            rnn.eval()
            val_sum_loss = []
            y_true_task = {}
            y_pred_task = {}
            y_pred_task_score = {}
            for index, tmp in enumerate(dataset_val):
                tmp_compound, tmp_y, tmp_smi = tmp
                loss = 0
                outputs = rnn(tmp_compound)
                # out_label = F.softmax(outputs, dim=1)
                # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
                # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i, x in enumerate(out_label.cpu().detach().numpy())]
                # y_pred.extend(pred)
                # y_pred_score.extend(pred_score)
                for i in range(tasks_num):
                    validId = np.where((tmp_y[:, i].cpu().numpy() == 0) | (tmp_y[:, i].cpu().numpy() == 1))[0]
                    if len(validId) == 0:
                        continue
                    y_pred = outputs[:, i * 2:(i + 1) * 2][torch.tensor(validId)].to(device)
                    y_label = tmp_y[:, i][torch.tensor(validId)].long().to(device)

                    # y_pred = torch.sigmoid(y_pred).view(-1)
                    # y_label = F.one_hot(y_label, 2).float().to(device)
                    loss += loss_function[i](y_pred, y_label)

                    pred_lable = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                    # pred_lable = np.zeros_like(y_pred.cpu().detach().numpy(), dtype=int)
                    # pred_lable[np.where(np.asarray(y_pred.cpu().detach().numpy()) > 0.5)] = 1
                    try:
                        y_true_task[i].extend(y_label.cpu().numpy())
                        y_pred_task[i].extend(pred_lable)
                        # y_pred_task_score[i].extend(y_pred)
                    except:
                        y_true_task[i] = []
                        y_pred_task[i] = []
                        # y_pred_task_score[i] = []
                        y_true_task[i].extend(y_label.cpu().numpy())
                        y_pred_task[i].extend(pred_lable)
                        # y_pred_task_score[i].extend(y_pred.cpu().detach().numpy())

                val_sum_loss.append(loss.cpu().detach().numpy())

            val_avg_loss = np.array(val_sum_loss).mean()

            trn_roc = [metrics.roc_auc_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
            trn_prc = [metrics.auc(precision_recall_curve(y_true_task[i], y_pred_task[i])[1],
                                   precision_recall_curve(y_true_task[i], y_pred_task[i])[0]) for i in
                       range(tasks_num)]
            # acc = [metrics.accuracy_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
            # recall = [metrics.recall_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
            # specificity = [cm[i][0, 0] / (cm[i][0, 0] + cm[i][0, 1]) for i in range(tasks_num)]

            print("epoch:", epoch, "   val  "  "avg_loss:", val_avg_loss,
                  # "acc: ", np.array(acc).mean(),
                  # "recall: ", np.array(recall).mean(),
                  # "specificity: ", np.array(specificity).mean(),
                  # " val_auc: ", trn_roc,
                  " val_auc: ", np.array(trn_roc).mean(),
                  # " val_pr: ", trn_prc,
                  " val_pr: ", np.array(trn_prc).mean())

            # 保存模型
            if val_avg_loss < test_best_loss:
                test_best_loss = val_avg_loss
                PATH = 'tox/lstm_net.pth'
                print("test save model")
                torch.save(rnn.state_dict(), PATH)

                with torch.no_grad():
                    rnn.eval()
                    test_sum_loss = []
                    y_true_task = {}
                    y_pred_task = {}
                    y_pred_task_score = {}
                    for index, tmp in enumerate(dataset_test):
                        tmp_compound, tmp_y, tmp_smi = tmp
                        loss = 0
                        outputs = rnn(tmp_compound)
                        # out_label = F.softmax(outputs, dim=1)
                        # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
                        # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i, x in enumerate(out_label.cpu().detach().numpy())]
                        # y_pred.extend(pred)
                        # y_pred_score.extend(pred_score)
                        for i in range(tasks_num):
                            validId = np.where((tmp_y[:, i].cpu().numpy() == 0) | (tmp_y[:, i].cpu().numpy() == 1))[0]
                            if len(validId) == 0:
                                continue
                            y_pred = outputs[:, i * 2:(i + 1) * 2][torch.tensor(validId)].to(device)
                            y_label = tmp_y[:, i][torch.tensor(validId)].long().to(device)

                            # y_pred = torch.sigmoid(y_pred).view(-1)
                            # y_label = F.one_hot(y_label, 2).float().to(device)
                            loss += loss_function[i](y_pred, y_label)

                            y_pred_s = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

                            pred_lable = np.zeros_like(y_pred_s, dtype=int)
                            pred_lable[np.where(np.asarray(y_pred_s) > 0.5)] = 1
                            try:
                                y_true_task[i].extend(y_label.cpu().numpy())
                                y_pred_task[i].extend(pred_lable)
                                y_pred_task_score[i].extend(y_pred_s)
                            except:
                                y_true_task[i] = []
                                y_pred_task[i] = []
                                y_pred_task_score[i] = []

                                y_true_task[i].extend(y_label.cpu().numpy())
                                y_pred_task[i].extend(pred_lable)
                                y_pred_task_score[i].extend(y_pred_s)

                        test_sum_loss.append(loss.cpu().detach().numpy())

                    trn_roc = [metrics.roc_auc_score(y_true_task[i], y_pred_task_score[i]) for i in range(tasks_num)]
                    trn_prc = [metrics.auc(precision_recall_curve(y_true_task[i], y_pred_task_score[i])[1],
                                           precision_recall_curve(y_true_task[i], y_pred_task_score[i])[0]) for i in
                               range(tasks_num)]
                    # print(len(trn_roc))
                    # print(sum(y_true_task[0]))
                    # print(sum(y_pred_task[0]))
                    acc = [metrics.accuracy_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
                    # recall = [metrics.recall_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
                    # specificity = [cm[i][0, 0] / (cm[i][0, 0] + cm[i][0, 1]) for i in range(tasks_num)]

                    print("epoch:", epoch, "   test  "  "avg_loss:", np.array(test_sum_loss).mean(),
                          "acc: ", np.array(acc).mean(),
                          # "recall: ", np.array(recall).mean(),
                          # "specificity: ", np.array(specificity).mean(),
                          # " test_auc: ", trn_roc,
                          " test_auc: ", np.array(trn_roc).mean(),
                          # " test_pr: ", trn_prc,
                          " test_pr: ", np.array(trn_prc).mean())


