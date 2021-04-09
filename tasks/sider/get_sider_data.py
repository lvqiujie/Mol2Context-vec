import pandas as pd

import numpy as np
from rdkit import Chem
import os
from rdkit.Chem import Descriptors
from sklearn.externals import joblib
# step 1
filepath="sider/sider.csv"
df = pd.read_csv(filepath, header=0, encoding="gbk")

w_file = open("sider/sider.smi", mode='w', encoding="utf-8")
all_label = []
all_smi = []
for line in df.values:
    # aa = np.array(line[:17], dtype = np.float64)
    # a =np.isnan(aa)
    smi = line[27].strip()
    all_label.append(line[0:27])
    all_smi.append(smi)
    w_file.write(smi + "\n")
    # mol = Chem.MolFromSmiles(smi)
    # try:
    #     if 12 <= Descriptors.MolWt(mol) <= 600:
    #         if -5 <= Descriptors.MolLogP(mol) <= 7:
    #             flag = True
    #             for t in mol.GetAtoms():
    #                 if t.GetSymbol() not in ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"]:
    #                     flag = False
    #                     print(t.GetSymbol())
    #                     break
    #             if flag:
    #                 all_label.append(line[0:27])
    #                 all_smi.append(smi)
    #                 w_file.write(smi + "\n")
    # except:
    #     print("error    "+smi)

w_file.close()

# step 2
adb = "mol2vec corpus -i sider/sider.smi -o sider/sider.cp -r 1 -j 4 --uncommon UNK --threshold 3"
d = os.popen(adb)
f = d.read()
print(f)


# step 3
vocab_path = "data/datasets/my_smi/smi_tran.vocab"
vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab_path).readlines()}

sentence_maxlen = 80

w_file = open("sider/sider_tran.cp_UNK", mode='w', encoding="utf-8")
label = []
smi = []
index = -1
mols_path = "sider/sider.cp_UNK"
mols_file = open(mols_path, mode='r',encoding="utf-8")
while True:
    line = mols_file.readline().strip()
    index += 1
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
    # print(token_ids)

    label.append(all_label[index])
    smi.append(all_smi[index])
    w_file.write(" ".join(str(i) for i in token_ids).strip()+"\n")
w_file.close()

joblib.dump(label, 'sider/label.pkl')
joblib.dump(smi, 'sider/smi.pkl')

# step 4
import os
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data import DATA_SET_DIR
from context_vec.smi_generator import SMIDataGenerator
from context_vec.smi_model import context_vec
import tensorflow as tf
from tensorflow import keras
from sklearn.externals import joblib

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'test_dataset': 'sider/sider_tran.cp_UNK',
    'vocab': 'my_smi/smi_tran.vocab',
    'model_dir': "smi_context_vec_512",
    'vocab_flag': False,
    'uncommon_threshold': 3,
    # 'vocab_size': 28914,
    # 'vocab_size': 748,
    'vocab_size': 13576,
    'num_sampled': 100,
    # 'charset_size': 262,
    'sentence_maxlen': 80,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 1000,
    'patience': 2,
    'batch_size': 512,
    'test_batch_size': 512,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': False,
    'n_lstm_layers': 2,
    'n_highway_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 512,
    'hidden_units_size': 300,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}

test_generator = SMIDataGenerator(parameters['test_dataset'],
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['test_batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

# Compile context_vec
context_vec_model = context_vec(parameters)
context_vec_model.compile_context_vec()

# context_vec_model.load(sampled_softmax=False)
#
# # Evaluate Bidirectional Language Model
# context_vec_model.evaluate(test_generator, parameters['test_batch_size'])
#
# # Build context_vec meta-model to deploy for production and persist in disk
# context_vec_model.wrap_multi_context_vec_encoder(print_summary=True)

# Load context_vec encoder
context_vec_model.load_context_vec_encoder()

# Get context_vec embeddings to feed as inputs for downstream tasks
context_vec_embeddings = context_vec_model.get_outputs(test_generator, output_type='word', state='all')
print(context_vec_embeddings.shape)

# 保存x
joblib.dump(context_vec_embeddings, 'sider/sider_embed.pkl')

