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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tasks.utils.util import *
from tasks.utils.model import *


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
    y = joblib.load('BBBP/label.pkl')

    x = joblib.load('BBBP/BBBP_embed.pkl')
    all_smi = joblib.load('BBBP/smi.pkl')
    print("data len is ",x.shape[0])

    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 5-Fold
    # 5-Fold
    train_split_x, train_split_y, train_split_smi, \
    val_split_x, val_split_y, val_split_smi, \
    test_split_x, test_split_y, test_split_smi, weights = split_data(x, y, all_smi, 4, "BBBP")

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    rnn = LSTM(1, task_type="sing", input_size=300, att=True).to(device)
    # 设置优化器和损失函数
    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay,
                                momentum = momentum)
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay)
    # loss_function = F.cross_entropy
    # loss_function = F.nll_loss
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCELoss()
    loss_function = nn.BCEWithLogitsLoss().to(device)
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

            # tmp_y = F.one_hot(tmp_y, 2).float().to(device)
            # print(label_one_hot)
            # aa = tmp_y.type(torch.FloatTensor).to(device)
            # bb = outputs
            # print(outputs.flatten())
            y_true.extend(tmp_y.cpu().numpy())
            # out_label = F.softmax(outputs, dim=1)
            # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
            # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i,x in enumerate(out_label.cpu().detach().numpy())]
            # y_pred.extend(pred)
            # y_pred_score.extend(pred_score)

            outputs = torch.sigmoid(outputs).view(-1)
            # tmp_y = F.one_hot(tmp_y, 2).float().to(device)
            loss = loss_function(outputs.to(device), tmp_y.float().to(device))

            pred = np.zeros_like(outputs.cpu().detach().numpy(), dtype=int)
            pred[np.where(np.asarray(outputs.cpu().detach().numpy()) > 0.5)] = 1
            y_pred.extend(pred)
            y_pred_score.extend(outputs.cpu().detach().numpy())
            # flood = (loss - b).abs() + b
            loss.backward()
            optimizer.step()

            sum_loss += loss
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        avg_loss = sum_loss / (index + 1)
        cm = metrics.confusion_matrix(y_true, y_pred)
        print("epoch:", epoch,"   train  "  "avg_loss:", avg_loss.item(),
                "acc: ", metrics.accuracy_score(y_true, y_pred),
                # "recall: ", metrics.recall_score(y_true, y_pred),
                # "specificity: ", cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                " train_auc: ", metrics.roc_auc_score(y_true, y_pred_score))

        # # 保存模型
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     PATH = 'BBBP/lstm_net.pth'
        #     print("train save model")
        #     torch.save(rnn.state_dict(), PATH)

        # 测试集部分换机器后再调试
        with torch.no_grad():
            rnn.eval()
            test_avg_loss = 0
            test_sum_loss = 0
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


                outputs = torch.sigmoid(outputs).view(-1)
                # tmp_y = F.one_hot(tmp_y, 2).float().to(device)
                loss = loss_function(outputs, tmp_y.float().to(device))
                # print(outputs.flatten())
                pred = np.zeros_like(outputs.cpu().detach().numpy(), dtype=int)
                pred[np.where(np.asarray(outputs.cpu().detach().numpy()) > 0.5)] = 1
                y_pred.extend(pred)
                y_pred_score.extend(outputs.cpu().detach().numpy())

                test_sum_loss += loss.item()

            test_avg_loss = test_sum_loss / (index + 1)
            cm = metrics.confusion_matrix(y_true, y_pred)
            print("epoch:", epoch,"   val  ", "avg_loss: ", test_avg_loss,
                  # "acc: ", metrics.accuracy_score(y_true, y_pred),
                  # "recall: ", metrics.recall_score(y_true, y_pred),
                  # "specificity: ", cm[0,0]/(cm[0,0]+cm[0,1]),
                  # "sensitivity: ", cm[1,1]/(cm[1,0]+cm[1,1]),
                  " test_auc: ", metrics.roc_auc_score(y_true, y_pred_score))
            # 保存模型
            if test_avg_loss < test_best_loss:
                test_best_loss = test_avg_loss
                PATH = 'BBBP/lstm_net.pth'
                print("test save model")
                torch.save(rnn.state_dict(), PATH)

                rnn.eval()
                test_avg_loss = 0
                test_sum_loss = 0
                y_true = []
                y_pred = []
                y_pred_score = []
                for index, tmp in enumerate(dataset_test):
                    tmp_compound, tmp_y, tmp_smi = tmp
                    loss = 0
                    y_true.extend(tmp_y.cpu().numpy())
                    outputs = rnn(tmp_compound)
                    # out_label = F.softmax(outputs, dim=1)
                    # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
                    # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i, x in enumerate(out_label.cpu().detach().numpy())]
                    # y_pred.extend(pred)
                    # y_pred_score.extend(pred_score)

                    outputs = torch.sigmoid(outputs).view(-1)
                    # tmp_y = F.one_hot(tmp_y, 2).float().to(device)
                    loss = loss_function(outputs, tmp_y.float().to(device))
                    # print(outputs.flatten())
                    pred = np.zeros_like(outputs.cpu().detach().numpy(), dtype=int)
                    pred[np.where(np.asarray(outputs.cpu().detach().numpy()) > 0.5)] = 1
                    y_pred.extend(pred)
                    y_pred_score.extend(outputs.cpu().detach().numpy())

                    test_sum_loss += loss.item()

                test_avg_loss = test_sum_loss / (index + 1)
                cm = metrics.confusion_matrix(y_true, y_pred)
                trn_prc = metrics.auc(precision_recall_curve(y_true, y_pred_score)[1],
                                      precision_recall_curve(y_true, y_pred_score)[0])
                print("epoch:", epoch, "   test  ", "avg_loss: ", test_avg_loss,
                      "acc: ", metrics.accuracy_score(y_true, y_pred),
                      # "recall: ", metrics.recall_score(y_true, y_pred),
                      # "specificity: ", cm[0, 0] / (cm[0, 0] + cm[0, 1]),
                      # "sensitivity: ", cm[1,1]/(cm[1,0]+cm[1,1]),
                      " test_auc: ", metrics.roc_auc_score(y_true, y_pred_score),
                      " test_pr: ", np.array(trn_prc).mean())