#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Train a LSTM network for sentiment analysis on IMDB review data.

Get the data from Kaggle: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

If choose to initialize the word embedding layer using Word2Vec, please make sure
to get the data GoogleNews-vectors-negative300.bin from:

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

Usage:

    python examples/imdb/train.py -f labeledTrainData.tsv -e 2 -eval 1 -s imdb.p \
        --vocab_file imdb.vocab [--use_w2v]

"""

from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from prepare import build_data_train  # noqa
from neon import logger as neon_logger  # noqa
from neon.backends import gen_backend  # noqa
from neon.data import ArrayIterator  # noqa
from neon.initializers import Uniform, GlorotUniform, Array  # noqa
from neon.layers import  (GeneralizedCost, LSTM, Affine, Dropout, LookupTable,
                         RecurrentSum, Recurrent, DeepBiLSTM, DeepBiRNN)  # noqa
from neon.models import Model  # noqa
from neon.optimizers import Adagrad  # noqa
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy, PrecisionRecall  # noqa
from neon.util.argparser import NeonArgparser, extract_valid_args  # noqa
from neon.util.compat import pickle  # noqa
from neon.callbacks.callbacks import Callbacks  # noqa
from neon.data.text_preprocessing import get_paddedXY  # noqa
import h5py  # noqa
import numpy as np
import os
import preprocessing


# Stop if validation error ever increases from epoch to epoch
def stop_func(s, v):
    if s is None:
        return (v, False)

    return (min(v, s), v > s)

def get_google_word2vec_W(fname, vocab, vocab_size=1000000, index_from=3):
    """
    Extract the embedding matrix from the given word2vec binary file and use this
    to initalize a new embedding matrix for words found in vocab.

    Conventions are to save indices for pad, oov, etc.:
    index 0: pad
    index 1: oov (or <unk>)
    index 2: <eos>. But often cases, the <eos> has already been in the
    preprocessed data, so no need to save an index for <eos>
    """
    f = open(fname, 'rb')
    header = f.readline()
    vocab1_size, embedding_dim = list(map(int, header.split()))
    binary_len = np.dtype('float32').itemsize * embedding_dim
    #vocab_size = min(len(vocab) + index_from, vocab_size)
    W = np.zeros((vocab_size, embedding_dim))

    found_words = {}
    for i, line in enumerate(range(vocab1_size)):
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = ''.join(word)
                break
            if ch != b'\n':
                word.append(ch.decode(encoding="ISO-8859-1"))
        if word in vocab:
            wrd_id = vocab[word] + index_from
            if wrd_id < vocab_size:
                W[wrd_id] = np.fromstring(
                    f.read(binary_len), dtype='float32')
                found_words[wrd_id] = 1
        else:
            f.read(binary_len)

    cnt = 0
    for wrd_id in range(vocab_size):
        if wrd_id not in found_words:
            W[wrd_id] = np.random.uniform(-0.25, 0.25, embedding_dim)
            cnt += 1
    assert cnt + len(found_words) == vocab_size

    f.close()

    return W, embedding_dim, vocab_size

# parse the command line arguments
parser = NeonArgparser(__doc__)

parser.add_argument('-onto', '--ontology',
                    default='dbpedia',
                    help='Replacement ontology')


parser.add_argument('-t', '--type',
                    default='generic',
                    help='Replacement strategy')

parser.add_argument('-n', '--clazz',
                    default=8,
                    help='Model to train 2 = Binary, 8 = Multiclass')

parser.add_argument('--rlayer_type', default='lstm',
                    choices=['bilstm', 'lstm', 'birnn', 'bibnrnn', 'rnn'],
                    help='type of recurrent layer to use (lstm, bilstm, rnn, birnn, bibnrnn)')

parser.add_argument('--vocab_file',
                    default='dbpedia_generic_train.tsv.vocab',
                    help='output file to save the processed vocabulary')

parser.add_argument('--use_w2v', action='store_true',
                    help='use downloaded Google Word2Vec')

parser.add_argument('--w2v',
                    default='/user/aedouard/home/Documents/_dev/event_detection/word_embedding/dbpedia_generic_merged.bin',
                    help='the pre-built Word2Vec')
args = parser.parse_args()


# hyperparameters
hidden_size = 128
embedding_dim = 128
vocab_size = 300000
sentence_length = 24
batch_size = 128
gradient_limit = 5
clip_gradients = True
num_epochs = args.epochs
embedding_update = True

def parseData(_file, valid=True):
    print(_file)

    #global nclass, train_set, valid_set
    h5f = h5py.File(_file, 'r')
    reviews, h5train, h5valid = h5f['reviews'], h5f['train'], h5f['valid']
    ntrain, nvalid, nclass = reviews.attrs[
                                 'ntrain'], reviews.attrs['nvalid'], reviews.attrs['nclass']
    # make train dataset
    Xy = h5train[:ntrain]
    X = [xy[1:] for xy in Xy]
    y = [xy[0] for xy in Xy]
    X_train, y_train = get_paddedXY(
        X, y, vocab_size=vocab_size, sentence_length=sentence_length)
    train_set = ArrayIterator(X_train, y_train, nclass=nclass)
    # make valid dataset
    if valid:
        Xy = h5valid[:nvalid]
        X = [xy[1:] for xy in Xy]
        y = [xy[0] for xy in Xy]
        X_valid, y_valid = get_paddedXY(
            X, y, vocab_size=vocab_size, sentence_length=sentence_length)
        valid_set = ArrayIterator(X_valid, y_valid, nclass=nclass)
        return reviews,train_set,valid_set
    else:
        return train_set

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))


train_file = "{}_{}_{}_train.tsv".format(args.ontology, args.type, args.clazz)
test_file = "{}_{}_{}_test.tsv".format(args.ontology, args.type, args.clazz)

print(train_file,test_file)


# get the preprocessed and tokenized data
train_file_h5, _, vocab = build_data_train(filepath=train_file,
                                              vocab_file="{}.vocab".format(train_file), skip_headers=False, train_ratio=0.9)
#parse the training file
reviews,train_set, valid_set = parseData(train_file_h5)
clazz = reviews.attrs['class_distribution']


#parse the test file
test_file_h5, fname_vocab, vocab = build_data_train(filepath=test_file,
                                   vocab_file="{}.vocab".format(test_file),
                                   skip_headers=False, train_ratio=1.0, clazz=clazz, vocab=vocab)

test_set= parseData(test_file_h5, valid=False)

neon_logger.display("Loading the Word2Vec vectors:")

# play around with google-news word vectors for init
if args.use_w2v:
    w2v_file = args.w2v
    vocab, rev_vocab = pickle.load(open(fname_vocab, 'rb'))
    init_emb_np, embedding_dim, vocab_size = get_google_word2vec_W(w2v_file, vocab,
                                                          vocab_size=vocab_size, index_from=3)
    neon_logger.display(
        "Done loading the Word2Vec vectors: embedding size - {} {}".format(embedding_dim,vocab_size))
    embedding_update = True
    init_emb = Array(val=be.array(init_emb_np))
else:
    init_emb = Uniform(-0.1 / embedding_dim, 0.1 / embedding_dim)

# initialization
g_uni = GlorotUniform()
rlayer = None

if args.rlayer_type == 'lstm':
    rlayer = LSTM(hidden_size, g_uni, activation=Tanh(),
                  gate_activation=Logistic(), reset_cells=True)
elif args.rlayer_type == 'bilstm':
    rlayer = DeepBiLSTM(hidden_size, g_uni, activation=Tanh(), depth=1,
                        gate_activation=Logistic(), reset_cells=True)
elif args.rlayer_type == 'rnn':
    rlayer = Recurrent(hidden_size, g_uni, activation=Tanh(), reset_cells=True)
elif args.rlayer_type == 'birnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(),
                       depth=1, reset_cells=True, batch_norm=False)
elif args.rlayer_type == 'bibnrnn':
    rlayer = DeepBiRNN(hidden_size, g_uni, activation=Tanh(),
                       depth=1, reset_cells=True, batch_norm=True)

if rlayer:
    # define layers
    layers = [
        LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb,
                    pad_idx=0, update=embedding_update),
        rlayer,
        RecurrentSum(),
        Dropout(keep=0.5),
        Affine(len(clazz), g_uni, bias=g_uni, activation=Softmax())
    ]

    # set the cost, metrics, optimizer
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
    metric = PrecisionRecall(num_classes=len(clazz))
    model = Model(layers=layers)
    optimizer = Adagrad(learning_rate=0.01)

    # configure callbacks
    callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
    # configure callbacks
    if args.callback_args['eval_freq'] is None:
        args.callback_args['eval_freq'] = 1
    callbacks.add_early_stop_callback(stop_func)
    callbacks.add_save_best_state_callback(os.path.join(args.data_dir, "{}_{}_best_state.pkl".format(args.ontology, args.type, args.clazz)))
    # train model
    model.fit(train_set,
              optimizer=optimizer,
              num_epochs=num_epochs,
              cost=cost,
              callbacks=callbacks)

    # eval model
    neon_logger.display("Train Accuracy - {}".format(100 * model.eval(train_set, metric=metric)))
    neon_logger.display("Valid Accuracy - {}".format(100 * model.eval(valid_set, metric=metric)))
    neon_logger.display("Test Accuracy - {}".format(100 * model.eval(test_set, metric=metric)))
else:
    print("No Layer provided...")