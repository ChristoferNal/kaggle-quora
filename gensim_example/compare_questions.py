from __future__ import absolute_import
from __future__ import print_function

import os
from multiprocessing import cpu_count

import gensim
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras.models import Model
from tqdm import tqdm

RS = 12357
ROUNDS = 800

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(input_dim):
    input = Input(shape=(input_dim,))
    dense1 = Dense(128)(input)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = merge([relu1, bn2], mode='sum')
    relu2 = Activation('relu')(res2)

    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = Merge(mode='sum')([relu2, bn3])
    relu3 = Activation('relu')(res3)

    feats = merge([relu3, relu2, relu1], mode='concat')
    bn4 = BatchNormalization()(feats)

    model = Model(input=input, output=bn4)

    return model


def create_network(input_dim):
    base_network = create_base_network(input_dim)
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
    return model


from keras.optimizers import Adam

df = pd.read_csv('../data/train.csv', )
tf = pd.read_csv('../data/test.csv')

number_of_train_samples = df.shape[0]

print(tf.head())

df = df.append(tf.loc[2000001:])

print(df.head())

df['question1'] = df['question1'].apply(lambda x: unicode(str(x), "utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x), "utf-8"))

tf['question1'] = tf['question1'].apply(lambda x: unicode(str(x),"utf-8"))
tf['question2'] = tf['question2'].apply(lambda x: unicode(str(x),"utf-8"))

cpus = cpu_count()

model_folder = '../data/3_word2vec.mdl'
google_model = '../data/GoogleNews-vectors-negative300.bin'

if os.path.isfile(google_model):
    print('Loading model GoogleNews-vectors-negative300...')
    model = gensim.models.KeyedVectors.load_word2vec_format(google_model, binary=True)
    print('GoogleNews-vectors-negative300 loaded successfully')
if os.path.isfile(model_folder):
    print('Model exists! Loading the model...')
    model = gensim.models.Word2Vec.load(model_folder)
else:
    questions = list(df['question1']) + list(df['question2'])
    c = 0
    for question in tqdm(questions):
        questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
        c += 1
    model = gensim.models.Word2Vec(questions, size=300, workers=cpus, iter=10, negative=20)
    model.init_sims(replace=True)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    print("Number of tokens in Word2Vec:", len(w2v.keys()))
    model.save('../data/3_word2vec.mdl')
    model.wv.save_word2vec_format('../data/3_word2vec.bin', binary=True)

def prepare_q_features(q1):
    tmp = []
    for question in tqdm(q1):
        if not question or not isinstance(question, basestring):
            mean_vec = mean_vec.mean(axis=0)
            tmp.append(mean_vec)
            continue
        q1_wordlist = question.split()
        mean_vec = np.zeros([len(q1_wordlist), 300])
        for _, word in enumerate(q1_wordlist):
            if not model.__contains__(word):
                continue
            v1 = model.wv.word_vec(word)
            mean_vec += v1
        mean_vec = mean_vec.mean(axis=0)
        tmp.append(mean_vec)
    return tmp

print('--------------------- Calculating Feature Vecs -----------------------------------')
q1 = df['question1'].tolist()
q2 = df['question2'].tolist()

tmp = prepare_q_features(q1)
df['q1_feats'] = list(tmp)
tmp = prepare_q_features(q2)
df['q2_feats'] = list(tmp)

num_train = df.shape[0]
num_test = df.shape[0] - num_train
print('--------------------------------------------------------')
print("| Number of training pairs: %i" % num_train)
print("| Number of testing pairs: %i" % num_test)
print('--------------------------------------------------------')

X_train = np.zeros([num_train, 2, 300])
Y_train = np.zeros([num_train])

b = [a[None, :] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

X_train[:, 0, :] = q1_feats[:num_train]
X_train[:, 1, :] = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values

X_test = np.zeros([num_test, 2, 300])
X_test[:, 0, :] = q1_feats[num_train:]
X_test[:, 1, :] = q2_feats[num_train:]

net = create_network(300)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

for epoch in range(10):
    net.fit([X_train[:, 0, :], X_train[:, 1, :]], Y_train,
            batch_size=128, nb_epoch=1, )
    probablity_testing = net.predict([X_test[:, 0, :], X_test[:, 1, :]], batch_size=128)
    # probablity_training = net.predict([X_train[:, 0, :], X_train[:, 1, :]], batch_size=128)
of = pd.Series(probablity_testing[:, 0], index=tf.values[1500001:2000001])
of.to_csv("word2vec_gensim_eucl_distance.csv", header=['is_duplicate'], index_label='test_id')
