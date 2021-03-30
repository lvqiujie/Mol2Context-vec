from rdkit import Chem
import torch
import os
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.externals import joblib
# from paper_data.plot_morgan import main
import numpy as np
import seaborn as sns
import math
import pickle
import random
from rdkit.Chem import MolFromSmiles
from AttentiveFP.Featurizer import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p_dropout = 0.2
fingerprint_dim = 200
# also known as l2_regularization_lambda
weight_decay = 5
learning_rate = 2.5
# for regression model
output_units_num = 1
radius = 2
T = 2
smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5]

class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]
class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(molgraph):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    #molgraph = graph_from_smiles_tuple(tuple(smiles))
    degrees = [0,1,2,3,4,5]
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array()}

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep
def gen_descriptor_data(smilesList):
    smiles_to_fingerprint_array = {}

    for i, smiles in enumerate(smilesList):
        #         if i > 5:
        #             print("Due to the limited computational resource, submission with more than 5 molecules will not be processed")
        #             break
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            molgraph = graph_from_smiles(smiles)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph)

            smiles_to_fingerprint_array[smiles] = arrayrep

        except:
            print(smiles,"%%%%%%%%")
            # time.sleep(3)
    return smiles_to_fingerprint_array

def save_smiles_dicts(smilesList, filename):
    # first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}

    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles, arrayrep in smiles_to_fingerprint_features.items():

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len, num_atom_features = atom_features.shape
        bond_len, num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    # then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}

    smiles_to_atom_mask = {}

    degrees = [0, 1, 2, 3, 4, 5]
    # then run through our numpy array again
    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))

        # get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len, num_atom_features))
        bonds = np.zeros((max_bond_len, num_bond_features))

        # then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len, len(degrees)))
        bond_neighbors = np.zeros((max_atom_len, len(degrees)))

        # now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        for i, feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j, feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i, degree_array in enumerate(atom_neighbors_list):
                    for j, value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count, j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i, degree_array in enumerate(bond_neighbors_list):
                    for j, value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count, j] = value
                    bond_neighbor_count += 1

        # then add everything to my arrays
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors

        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    feature_dicts = {}
    #     feature_dicts['smiles_to_atom_mask'] = smiles_to_atom_mask
    #     feature_dicts['smiles_to_atom_info']= smiles_to_atom_info
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }
    pickle.dump(feature_dicts, open(filename + '.pickle', "wb"))
    print('feature dicts file saved as ' + filename + '.pickle')
    return feature_dicts



def split_data(x, y, all_smi, lens, k_fold):
    y = np.array(y, dtype=np.float64)
    all_smi = np.array(all_smi)
    lens = np.array(lens)

    # save_path = 'lipop/'+str(k_fold)+'-fold-index.pkl'
    # if os.path.isfile(save_path):
    #     index = joblib.load(save_path)
    #     train_split_x = x[index["train_index"]]
    #     train_split_y = y[index["train_index"]]
    #     val_split_x = x[index["val_index"]]
    #     val_split_y = y[index["val_index"]]
    #     test_split_x = x[index["test_index"]]
    #     test_split_y = y[index["test_index"]]
    #     train_weights = joblib.load('lipop/train_weights.pkl')
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
                     }, 'lipop/'+str(i+1)+'-fold-index.pkl')
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

def get_smiles_array(smilesList, feature_dicts):
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    for smiles in smilesList:
        x_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        x_atom.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        x_atom_index.append(feature_dicts['smiles_to_atom_neighbors'][smiles])
        x_bond_index.append(feature_dicts['smiles_to_bond_neighbors'][smiles])
    return np.asarray(x_atom),np.asarray(x_bonds),np.asarray(x_atom_index),\
        np.asarray(x_bond_index),np.asarray(x_mask),feature_dicts['smiles_to_rdkit_list']

class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, \
                 fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
        #         self.mol_GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for t in range(T)])
        #         self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for t in range(T)])
        #         self.mol_attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for t in range(T)])

        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        atom_mask = atom_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[softmax_mask == mol_length - 1] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num,
                                                                fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        #             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        #             print(attention_weight)
        attention_weight = attention_weight * attend_mask
        #         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        #             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        #             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        # do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius - 1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]

            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num,
                                                                          fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            #             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            #             print(attention_weight)
            neighbor_feature_transform = self.attend[d + 1](self.dropout(neighbor_feature))
            #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            #             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d + 1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)

        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)

        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask
            #             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight, activated_features_transform), -2)
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            #             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)

        mol_prediction = self.output(self.dropout(mol_feature))

        return atom_feature, mol_prediction, mol_feature

class LSTM(nn.Module):
    """搭建rnn网络"""
    def __init__(self, model):
        super(LSTM, self).__init__()

        self.matrix = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        self.model = model
        self.fc = nn.Linear(600, 1024)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,)
        #
        # # self.fc1 = nn.Linear(512, 1024)
        # # self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512 + 200, 1)

        # self.fc5 = nn.Linear(200, 1)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x, x_lens, tmp_smi):
        # print(self.matrix1, self.matrix2, self.matrix3)
        # bs = len(x)
        # length = np.array([t.shape[0] for t in x])

        x = x.to(device)
        x = self.matrix[0] * x[:, 0, :, :] + self.matrix[1] * x[:, 1, :, :] + self.matrix[2] * x[:, 2, :, :]
        #
        x = self.fc(x.to(device)).to(device)
        # packing
        # embed_packed = pack_padded_sequence(x, x_lens,
        #                                     batch_first=True,
        #                                     enforce_sorted=False)

        out, (hidden, cell) = self.lstm(x)     #h_state是之前的隐层状态

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(tmp_smi,
                                                                                                     feature_dicts)
        atoms_prediction, mol_prediction, mol_feature = self.model(torch.Tensor(x_atom).to(device),
                                                                    torch.Tensor(x_bonds).to(device),
                                                                    torch.cuda.LongTensor(x_atom_index),
                                                                    torch.cuda.LongTensor(x_bond_index),
                                                                    torch.Tensor(x_mask).to(device))

        # unpacking
        # out, lens = pad_packed_sequence(out, batch_first=True)
        alpha_n =0
        att =0
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
        out_tmp = torch.cat((out_tmp.view(-1, 512), mol_feature.view(-1, 200)), dim=1)
        out_tmp = self.fc4(out_tmp)
        # out_tmp = self.fc5(mol_feature)
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
        return out_tmp, alpha_n, att

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
    # 设置超参数
    input_size = 512
    num_layers = 2  # 定义超参数rnn的层数，层数为1层
    hidden_size = 512  # 定义超参数rnn的循环神经元个数，个数为32个
    learning_rate = 0.01  # 定义超参数学习率
    epoch_num = 1000
    batch_size = 64
    best_loss = 100000
    test_best_loss = 100000
    weight_decay = 1e-5
    momentum = 0.9

    b = 0.04
    seed = 188
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # filepath = "lipop/delaney.csv"
    # df = pd.read_csv(filepath, header=0, encoding="gbk")
    y = joblib.load('lipop/label.pkl')
    all_smi = np.array(joblib.load('lipop/smi.pkl'))

    x = joblib.load('lipop/lipop_embed.pkl')
    lens = joblib.load('lipop/lens.pkl')

    # 5-Fold
    train_split_x, train_split_y, train_split_smi, train_split_lens,\
    val_split_x, val_split_y, val_split_smi, val_split_lens,\
    test_split_x, test_split_y, test_split_smi, test_split_lens = split_data(x, y, all_smi, lens, 3)

    data_train = MyDataset(train_split_x, train_split_y, train_split_smi, train_split_lens)
    dataset_train = data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

    data_val = MyDataset(val_split_x, val_split_y, val_split_smi, val_split_lens)
    dataset_val = data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    data_test = MyDataset(test_split_x, test_split_y, test_split_smi, test_split_lens)
    dataset_test = data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    data_all = MyDataset(x, y, all_smi, lens)
    dataset_all = data.DataLoader(dataset=data_all, batch_size=1, shuffle=True)

    raw_filename = "lipop/Lipophilicity.csv"
    feature_filename = raw_filename.replace('.csv','.pickle')
    filename = raw_filename.replace('.csv','')
    prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles: ", len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print(smiles,"######3")
            pass
    feature_filename = 'lipop/Lipophilicity'
    # if os.path.isfile(feature_filename):
    #     print("NO lipop/delaney-processed.pickle")
    #     feature_dicts = pickle.load(open(feature_filename, "rb"))
    # else:
    feature_dicts = save_smiles_dicts(smilesList, feature_filename)

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
        [canonical_smiles_list[0]], feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                        fingerprint_dim, output_units_num, p_dropout)
    model.to(device)
    rnn = LSTM(model).to(device)


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

            # x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(tmp_smi,
            #                                                                                              feature_dicts)
            # atoms_prediction, outputs, mol_feature = rnn(torch.Tensor(x_atom).to(device),
            #                                                            torch.Tensor(x_bonds).to(device),
            #                                                            torch.cuda.LongTensor(x_atom_index),
            #                                                            torch.cuda.LongTensor(x_bond_index),
            #                                                            torch.Tensor(x_mask).to(device))

            outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device), tmp_smi)
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
        #     PATH = 'lipop/lstm_net.pth'
        #     print("train save model")
        #     torch.save(rnn.state_dict(), PATH)

        # print(task_matrix[0], task_matrix[1], task_matrix[2])
        with torch.no_grad():
            rnn.eval()
            test_avg_loss = 0
            test_sum_loss = 0
            for index, tmp in enumerate(dataset_val):
                tmp_compound, tmp_y, tmp_smi, tmp_len = tmp

                outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device), tmp_smi)
                # print(outputs.flatten())
                loss = loss_function(outputs.flatten(), tmp_y.type(torch.FloatTensor).to(device))
                test_sum_loss += loss.item()


            test_avg_loss = test_sum_loss / (index + 1)
            print("epoch:", epoch,"   val  ", "avg_loss: ", test_avg_loss)
            # 保存模型
            if test_avg_loss < test_best_loss:
                test_best_loss = test_avg_loss
                print("test save model")
                torch.save(rnn.state_dict(), 'lipop/lstm_net.pth')

                rnn.eval()
                test_avg_loss = 0
                test_sum_loss = 0
                all_pred = []
                all_label = []

                for index, tmp in enumerate(dataset_test):
                    tmp_compound, tmp_y, tmp_smi, tmp_len = tmp
                    loss = 0
                    outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device),tmp_smi)

                    y_pred = outputs.to(device).view(-1)
                    y_label = tmp_y.float().to(device).view(-1)

                    all_label.extend(y_label.cpu().numpy())
                    all_pred.extend(y_pred.cpu().numpy())

                    # y_pred = torch.sigmoid(y_pred).view(-1)
                    # y_label = F.one_hot(y_label, 2).float().to(device)
                    loss += loss_function(y_pred, y_label)

                    test_sum_loss += loss.item()


                mse = mean_squared_error(all_label, all_pred)
                mae = mean_absolute_error(all_label, all_pred)
                rmse = np.sqrt(mse)
                test_avg_loss = test_sum_loss / (index + 1)

                print("epoch:", epoch, "   test   avg_loss:", test_avg_loss
                      ," mae : ", mae
                      ," rmse : ", rmse)
                # if rmse < 0.7:
                #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #
                #     rnn.eval()
                #     for index, tmp in enumerate(dataset_all):
                #         tmp_compound, tmp_y, tmp_smi, tmp_len = tmp
                #         outputs, alpha_n, att_n = rnn(tmp_compound.to(device), tmp_len.to(device), tmp_smi)
                #         print(outputs.cpu().detach().numpy()[0][0], tmp_y.cpu().detach().numpy()[0], tmp_smi[0])

