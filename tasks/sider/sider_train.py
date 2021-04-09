import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from tasks.utils.util import *
from tasks.utils.model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 设置超参数
    input_size = 512
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.001  # 定义超参数学习率
    epoch_num = 200
    batch_size = 32
    best_loss = 10000
    test_best_loss = 10000
    weight_decay = 1e-5
    momentum = 0.9
    # b = 0.6

    seed = 199
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    y = joblib.load('sider/label.pkl')
    all_smi = joblib.load('sider/smi.pkl')

    x = joblib.load('sider/sider_embed.pkl')
    print("data len is ", x.shape[0])
    tasks_num = 27

    # 5-Fold
    train_split_x, train_split_y, train_split_smi, \
    val_split_x, val_split_y, val_split_smi, \
    test_split_x, test_split_y, test_split_smi, weights = split_multi_label(x, y, all_smi, 1, 'sider')

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    rnn = LSTM(tasks_num, task_type="muti", input_size=600).to(device)
    # 设置优化器和损失函数
    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay,
    #                             momentum = momentum)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # loss_function = F.cross_entropy
    # loss_function = F.nll_loss
    # loss_function = nn.CrossEntropyLoss()
    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight).to(device), reduction='mean') for weight in weights]
    # loss_function = nn.BCELoss()
    # loss_function = nn.BCEWithLogitsLoss()
    # loss_function = FocalLoss(alpha=1 / train_weights[0])

    # 按照以下的过程进行参数的训练
    for epoch in range(epoch_num):
        sum_loss = []
        rnn.train()
        y_true_task = {}
        y_pred_task = {}
        y_pred_task_score = {}
        for index, tmp in enumerate(dataset_train):
            tmp_compound, tmp_y, tmp_smi = tmp
            optimizer.zero_grad()
            outputs = rnn(tmp_compound.to(device))
            loss = 0

            for i in range(tasks_num):
                validId = np.where((tmp_y[:,i].cpu().numpy() == 0) | (tmp_y[:,i].cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                # print(outputs.shape)
                y_pred = outputs[:, i * 2:(i + 1) * 2][torch.tensor(validId)].to(device)
                y_label = tmp_y[:,i][torch.tensor(validId)].long().to(device)

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

            # loss = (loss - b).abs() + b
            loss.backward()
            optimizer.step()

            sum_loss.append(loss.cpu().detach().numpy())
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        # acc = [metrics.accuracy_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
        # recall = [metrics.recall_score(y_true_task[i], y_pred_task[i]) for i in range(tasks_num)]
        # specificity = [cm[i][0, 0] / (cm[i][0, 0] + cm[i][0, 1]) for i in range(tasks_num)]

        print("epoch:", epoch,"   train  "  "avg_loss:", np.array(sum_loss).mean())

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
                PATH = 'sider/lstm_net.pth'
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


