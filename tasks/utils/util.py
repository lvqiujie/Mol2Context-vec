import sys
sys.path.append('./')
import numpy as np
from sklearn.model_selection import KFold
from sklearn.externals import joblib


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


def split_multi_label(x, y, smi, k_fold, name):
    y = np.array(y).astype(float)
    # y[np.where(np.isnan(y))] = 6
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
