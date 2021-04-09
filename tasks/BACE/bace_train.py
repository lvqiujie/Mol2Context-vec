# from rdkit import Chem
# from rdkit.Chem import AllChem
import random
from tasks.utils.model import *
from sklearn.externals import joblib
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_len(smi):
    mol = Chem.MolFromSmiles(smi)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    return len(mol_atoms)


if __name__ == '__main__':
    # 设置超参数
    input_size = 512
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.01  # 定义超参数学习率
    epoch_num = 2000
    batch_size = 32
    best_loss = 10000
    test_best_loss = 10000
    weight_decay = 1e-5
    momentum = 0.9

    # b = 0.2
    all_smi = np.array(joblib.load('bace/smi.pkl'))
    y = joblib.load('bace/label.pkl')

    x = joblib.load('bace/bace_embed.pkl')
    print("data len is ",x.shape[0])

    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 5-Fold
    train_split_x, train_split_y, train_split_smi, \
    val_split_x, val_split_y, val_split_smi, \
    test_split_x, test_split_y, test_split_smi, weights = split_data(x, y, all_smi, 3, "bace")

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    rnn = LSTM(1, task_type="sing", input_size=300, att=True).to(device)


    # 设置优化器和损失函数
    #使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
    optimizer = torch.optim.SGD(rnn.parameters(),
                                lr=learning_rate, weight_decay=weight_decay, momentum = momentum)
    # optimizer = torch.optim.Adam(list(rnn.parameters())+[matrix1, matrix2, matrix3],
    #                              lr=learning_rate, weight_decay = weight_decay)
    # optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # loss_function = F.cross_entropy
    # loss_function = F.nll_loss
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCELoss()
    loss_function = nn.BCEWithLogitsLoss().to(device)
    # loss_function = FocalLoss(alpha=1 / train_weights[0])
    # loss_function = torch.nn.CrossEntropyLoss(torch.Tensor(train_weights).to(device), reduction='mean')

    # 按照以下的过程进行参数的训练
    for epoch in range(epoch_num):
        avg_loss = 0
        sum_loss = 0
        rnn.train()
        y_true_task = []
        y_pred_task = []
        y_pred_task_score = []
        for index, tmp in enumerate(dataset_train):
            tmp_compound, tmp_y, tmp_smi = tmp
            # aa = get_delete(tmp_smi[0])
            optimizer.zero_grad()
            outputs = rnn(tmp_compound)
            loss = 0
            # tmp_y = F.one_hot(tmp_y, 2).float().to(device)
            # print(label_one_hot)
            # aa = tmp_y.type(torch.FloatTensor).to(device)
            # bb = outputs
            # print(outputs.flatten())
            # out_label = F.softmax(outputs, dim=1)
            # pred = out_label.data.max(1, keepdim=True)[1].view(-1).cpu().numpy()
            # pred_score = [x[tmp_y.cpu().detach().numpy()[i]] for i,x in enumerate(out_label.cpu().detach().numpy())]
            # y_pred.extend(pred)
            # y_pred_score.extend(pred_score)

            y_pred = outputs.to(device).view(-1)
            y_label = tmp_y.float().to(device).view(-1)


            # y_label = F.one_hot(y_label, 2).float().to(device)
            loss += loss_function(y_pred, y_label)

            # pred_lable = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

            y_pred = torch.sigmoid(y_pred.detach().cpu()).view(-1).numpy()
            pred_lable = np.zeros_like(y_pred, dtype=int)
            pred_lable[np.where(np.asarray(y_pred) > 0.5)] = 1

            y_true_task.extend(y_label.cpu().numpy())
            y_pred_task.extend(pred_lable)
            y_pred_task_score.extend(y_pred)

            # loss = (loss - b).abs() + b
            loss.backward()
            optimizer.step()

            sum_loss += loss
            # print("epoch:", epoch, "index: ", index,"loss:", loss.item())
        avg_loss = sum_loss / (index + 1)
        # cm = [metrics.confusion_matrix(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        trn_roc = metrics.roc_auc_score(y_true_task, y_pred_task_score)
        trn_prc = metrics.auc(precision_recall_curve(y_true_task, y_pred_task_score)[1],
                               precision_recall_curve(y_true_task, y_pred_task_score)[0])
        # acc = [metrics.accuracy_score(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        # recall = [metrics.recall_score(y_true_task[i], y_pred_task[i]) for i in range(len(tasks))]
        # specificity = [cm[i][0, 0] / (cm[i][0, 0] + cm[i][0, 1]) for i in range(len(tasks))]

        print("epoch:", epoch,"   train  "  "avg_loss:", avg_loss.item(),
                # "acc: ", np.array(acc).mean(),
                # "recall: ", np.array(recall).mean(),
                # "specificity: ", np.array(specificity).mean(),
                " train_auc: ", np.array(trn_roc).mean(),
                " train_pr: ", np.array(trn_prc).mean())

        with torch.no_grad():
            rnn.eval()
            test_avg_loss = 0
            test_sum_loss = 0
            y_true_task = []
            y_pred_task = []
            y_pred_task_score = []
            for index, tmp in enumerate(dataset_val):
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

            print("epoch:", epoch, "   val   avg_loss:", test_avg_loss,
                  # "acc: ", np.array(acc).mean(),
                  # "recall: ", np.array(recall).mean(),
                  # "specificity: ", np.array(specificity).mean(),
                  " test_auc: ", np.array(trn_roc).mean(),
                  " test_pr: ", np.array(trn_prc).mean())

            if test_avg_loss < test_best_loss:
                test_best_loss = test_avg_loss
                PATH = 'bace/lstm_net.pth'
                print("test save model")
                torch.save(rnn.state_dict(), PATH)
                att_flag = False
                # if test_avg_loss < 0.6:
                #     att_flag = True
                # print(matrix1, matrix2, matrix3)
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

                        if att_flag:
                            att = alpha_n.cpu().detach().numpy()
                            for att_i in range(alpha_n.shape[0]):
                                smi_len = get_len(tmp_smi[att_i])
                                if smi_len > 40:
                                    continue
                                att_tmp = att[att_i,:smi_len*2,:smi_len*2]
                                att_heatmap = att_tmp[1::2, 1::2]
                                att_heatmap = (att_heatmap - att_heatmap.min()) / (att_heatmap.max() - att_heatmap.min())
                                # f, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=1)

                                fig = sns.heatmap(att_heatmap, cmap='OrRd')
                                # plt.show()
                                scatter_fig = fig.get_figure()
                                try:
                                    scatter_fig.savefig("bace/att_img/"+str(tmp_smi[att_i])+".png", dpi=400)
                                except:
                                    continue
                                finally:
                                    plt.close()

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
                          "recall: ", np.array(recall).mean(),
                          "specificity: ", np.array(specificity).mean(),
                          " test_auc: ", np.array(trn_roc).mean(),
                          " test_pr: ", np.array(trn_prc).mean())

