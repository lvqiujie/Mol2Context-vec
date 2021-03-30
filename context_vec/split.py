import numpy as np
import os
import random


mols_src_all = []
all_dict = {}
with open("data/datasets/my_smi_0/mols.cp_UNK") as fp:
    for i, line in enumerate(fp):
        mols_src_all.append(line)
        for j, token in enumerate(line.split()):
            token = token.strip()
            if token in all_dict:
                all_dict[token] += 1
            else:
                all_dict[token] = 1

f = open("data/datasets/my_smi_0/smi_tran.vocab", 'w')
if "UNK" in all_dict:
    all_dict.pop('UNK','404')
f.write('<pad> 0' + "\n")
f.write('<unk> 1' + "\n")
f.write('unk 1' + "\n")
f.write('<bos> 2' + "\n")
f.write('<eos> 3' + "\n")
index = 4
for k, v in all_dict.items():
    if v <= 3:
        continue
    f.write(k+' ' +str(index)+"\n")
    index += 1
f.close()

vocab_path = "data/datasets/my_smi_0/smi_tran.vocab"
vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab_path).readlines()}


w_file = open("data/datasets/my_smi_0/mols_tran.cp_UNK", mode='w', encoding="utf-8")
sentence_maxlen = 80
for line in mols_src_all:
    if "None".__eq__(line.strip()) or "UNK".__eq__(line.strip()):
        continue
    if not line:
        break
    token_ids = np.zeros((sentence_maxlen,), dtype=np.int64)
    # Add begin of sentence index
    token_ids[0] = vocab['<bos>']
    for j, token in enumerate(line.split()[:sentence_maxlen - 2]):
        # print(token)
        if token.lower() in vocab:
            token_ids[j + 1] = vocab[token.lower()]
        else:
            token_ids[j + 1] = vocab['<unk>']
    # Add end of sentence index
    if token_ids[1]:
        token_ids[j + 2] = vocab['<eos>']
    w_file.write(" ".join(str(i) for i in token_ids).strip()+"\n")
w_file.close()

mols_path = "data/my_smi_0/mols_tran.cp_UNK"
mols_file = open(mols_path, mode='r',encoding="utf-8")
mols_all = mols_file.readlines()
all_num = len(mols_all)

random.shuffle(mols_all)
train_num = all_num * 0.8
w_file = open("./data/my_smi_0/mols_train.cp_UNK", mode='w', encoding="utf-8")
mols_train = mols_all[:int(train_num)]
for i in mols_train:
    w_file.write(str(i).strip()+"\n")
w_file.close()

val_num = all_num * 0.9
w_file = open("./data/my_smi_0/mols_val.cp_UNK", mode='w', encoding="utf-8")
mols_train = mols_all[int(train_num):int(val_num)]
for i in mols_train:
    w_file.write(str(i).strip()+"\n")
w_file.close()

w_file = open("./data/datasets/my_smi_0/mols_test.cp_UNK", mode='w', encoding="utf-8")
mols_train = mols_all[int(val_num):]
for i in mols_train:
    w_file.write(str(i).strip()+"\n")
w_file.close()

