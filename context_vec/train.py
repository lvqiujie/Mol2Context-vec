import os
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from context_vec.smi_generator import SMIDataGenerator
from context_vec.smi_model import Context_vec
import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
keras.backend.set_session(sess)
DATA_SET_DIR = "./context_vec/data/"
my_smi = "my_smi_0"
parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': my_smi+'/mols_train.cp_UNK',
    'valid_dataset': my_smi+'/mols_val.cp_UNK',
    'test_dataset': my_smi+'/mols_test.cp_UNK',
    'vocab': my_smi+'/smi_tran.vocab',
    'vocab_flag': False,
    'model_dir': "smi_context_vec_0_300",
    'uncommon_threshold': 3,
    # 'vocab_size': 28914,
    # 'vocab_size': 748,
    # 'vocab_size': 121,
    'vocab_size': 13576,
    'num_sampled': 100,
    # 'charset_size': 262,
    'sentence_maxlen': 80,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 50,
    'patience': 2,
    'batch_size': 512,
    'test_batch_size': 512,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
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
    'hidden_units_size': 300,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}

def vocab_generator(corpus):
    all_dict = {}
    with open(corpus) as fp:
        for i, line in enumerate(fp):
            for j, token in enumerate(line.split()):
                token = token.strip()
                if token in all_dict:
                    all_dict[token] += 1
                else:
                    all_dict[token] = 1

    f = open(os.path.join(DATA_SET_DIR, my_smi+"/smi_tran.vocab"), 'w')
    if "UNK" in all_dict:
        all_dict.pop('UNK','404')
    f.write('<pad> 0' + "\n")
    f.write('<unk> 1' + "\n")
    f.write('unk 1' + "\n")
    f.write('<bos> 2' + "\n")
    f.write('<eos> 3' + "\n")
    index = 4
    for k, v in all_dict.items():
        if v <= parameters['uncommon_threshold']:
            continue
        f.write(k+' ' +str(index)+"\n")
        index += 1
    f.close()

print(parameters)
if not os.path.exists("context_vec/models/"+parameters['model_dir']):
    os.mkdir("context_vec/models/"+parameters['model_dir'])

if parameters['vocab_flag']:
    print("-------------        vocab         -------------")
    vocab_generator(os.path.join(DATA_SET_DIR, my_smi+"/mols.cp_UNK"))
# Set-up Generators
train_generator = SMIDataGenerator(os.path.join(DATA_SET_DIR, parameters['train_dataset']),
                                  os.path.join(DATA_SET_DIR, parameters['vocab']),
                                  sentence_maxlen=parameters['sentence_maxlen'],
                                  token_maxlen=parameters['token_maxlen'],
                                  batch_size=parameters['batch_size'],
                                  shuffle=parameters['shuffle'],
                                  token_encoding=parameters['token_encoding'])

val_generator = SMIDataGenerator(os.path.join(DATA_SET_DIR, parameters['valid_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

test_generator = SMIDataGenerator(os.path.join(DATA_SET_DIR, parameters['test_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['test_batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

# Compile context_vec
context_vec_model = Context_vec(parameters)
context_vec_model.compile_context_vec(print_summary=True)

# Train context_vec
# context_vec_model.train(train_data=train_generator, valid_data=val_generator)

# Persist context_vec Bidirectional Language Model in disk
context_vec_model.save(sampled_softmax=False, model_dir=parameters['model_dir'])

# Evaluate Bidirectional Language Model
context_vec_model.evaluate(test_generator, parameters['test_batch_size'])

# Build context_vec meta-model to deploy for production and persist in disk
context_vec_model.wrap_multi_context_vec_encoder(print_summary=True, save=True)

# Load context_vec encoder
context_vec_model.load_context_vec_encoder()

# Get context_vec embeddings to feed as inputs for downstream tasks
# context_vec_embeddings = context_vec_model.get_outputs(test_generator, output_type='word', state='mean')

# BUILD & TRAIN NEW KERAS MODEL FOR DOWNSTREAM TASK (E.G., TEXT CLASSIFICATION)
a = 1