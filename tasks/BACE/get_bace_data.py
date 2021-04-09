import pandas as pd
from sklearn.externals import joblib
import numpy as np
import os

# step 1
filepath="bace/bace.csv"
df = pd.read_csv(filepath, header=0, encoding="gbk")

w_file = open("bace/bace.smi", mode='w', encoding="utf-8")
all_label = []
all_smi = []
for line in df.values:
    # aa = np.array(line[:17], dtype = np.float64)
    # a =np.isnan(aa)
    all_label.append(line[2])
    all_smi.append(line[0])

    w_file.write(line[0]+"\n")
w_file.close()

# step 2
adb = "mol2vec corpus -i bace/bace.smi -o bace/bace.cp -r 1 -j 4 --uncommon UNK --threshold 3"
d = os.popen(adb)
f = d.read()
print(f)


# step 3
vocab_path = "data/datasets/my_smi_0/smi_tran.vocab"
vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab_path).readlines()}

sentence_maxlen = 80

w_file = open("bace/bace_tran.cp_UNK", mode='w', encoding="utf-8")
label = []
smi = []
index = -1
mols_path = "bace/bace.cp_UNK"
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

joblib.dump(label, 'bace/label.pkl')
joblib.dump(smi, 'bace/smi.pkl')

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
    'test_dataset': 'bace/bace_tran.cp_UNK',
    'vocab': 'my_smi_0/smi_tran.vocab',
    'model_dir': "smi_elmo_best",
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
    'lstm_units_size': 300,
    'hidden_units_size': 150,
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
elmo_model.compile_context_vec()

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
joblib.dump(elmo_embeddings, 'bace/bace_embed.pkl')

