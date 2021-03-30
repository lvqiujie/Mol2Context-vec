import pandas as pd
from sklearn.externals import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# step 1
filepath="esol/delaney-processed3.csv"
df = pd.read_csv(filepath, header=0, encoding="gbk")

w_file = open("esol/esol.smi", mode='w',encoding="utf-8")
all_label = []
all_smi = []
for line in df.values:
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(line[9].strip()), isomericSmiles=True)
    if "Cc1cccc([N+](=O)[O-])c1".__eq__(smi):
        ss= 1
    if len(smi) <= 0:
        break
    mol = Chem.MolFromSmiles(smi)
    all_smi.append(smi)
    all_label.append(line[8])
    w_file.write(smi + "\n")
    # try:
    #     if 12 <= Descriptors.MolWt(mol) <= 600:
    #         if -5 <= Descriptors.MolLogP(mol) <= 7:
    #             flag = True
    #             for t in mol.GetAtoms():
    #                 if t.GetSymbol() not in ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"]:
    #                     flag = False
    #                     print("############    ",smi,t.GetSymbol())
    #                     break
    #             if flag:
    #                 all_smi.append(smi)
    #                 all_label.append(line[8])
    #                 w_file.write(smi + "\n")
    # except:
    #     print("error    "+smi)
w_file.close()


# step 2
adb = "mol2vec corpus -i esol/esol.smi -o esol/esol.cp -r 1 -j 4 --uncommon UNK --threshold 3"
d = os.popen(adb)
f = d.read()
print(f)


# step 3
vocab_path = "data/datasets/my_smi/smi_tran.vocab"
vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab_path).readlines()}

sentence_maxlen = 80

w_file = open("esol/esol_tran.cp_UNK", mode='w', encoding="utf-8")
label = []
smi = []
lens = []
index = -1
mols_path = "esol/esol.cp_UNK"
mols_file = open(mols_path, mode='r',encoding="utf-8")
while True:
    line = mols_file.readline().strip()
    words = line.split()
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
    lens.append(len(words) if len(words) + 2 <= sentence_maxlen else 80)
    w_file.write(" ".join(str(i) for i in token_ids).strip()+"\n")
w_file.close()
joblib.dump(label, 'esol/label.pkl')
joblib.dump(smi, 'esol/smi.pkl')
joblib.dump(lens, 'esol/lens.pkl')


# step 4
import os
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data import DATA_SET_DIR
from elmo.smi_generator import SMIDataGenerator
from elmo.smi_model import ELMo
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
    'test_dataset': 'esol/esol_tran.cp_UNK',
    'vocab': 'my_smi/smi_tran.vocab',
    'model_dir': "smi_elmo_512",
    'vocab_flag': False,
    'uncommon_threshold': 3,
    # 'vocab_size': 28914,
    # 'vocab_size': 748,
    'vocab_size': 13576,
    # 'vocab_size': 121,
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

# Compile ELMo
elmo_model = ELMo(parameters)
elmo_model.compile_elmo()

# elmo_model.load(sampled_softmax=False)
#
# # Evaluate Bidirectional Language Model
# elmo_model.evaluate(test_generator, parameters['test_batch_size'])
#
# # Build ELMo meta-model to deploy for production and persist in disk
# elmo_model.wrap_multi_elmo_encoder(print_summary=True)

# Load ELMo encoder
elmo_model.load_elmo_encoder()

# Get ELMo embeddings to feed as inputs for downstream tasks
elmo_embeddings = elmo_model.get_outputs(test_generator, output_type='word', state='all')
print(elmo_embeddings.shape)

# 保存x
joblib.dump(elmo_embeddings, 'esol/esol_embed.pkl')

