import os.path
import sys
from multiprocessing import cpu_count

import gensim
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from keras.layers import Input, Dense, concatenate
from keras.models import Model

reload(sys)
sys.setdefaultencoding('utf8')


RS = 12357
EMBEDDING_SIZE = 300
np.random.seed(RS)
input_folder = '../data/'
model_folder = '../tmp/model_with_train_test_data'
cpus = cpu_count()


def read_data():
    df_train = pd.read_csv(input_folder + 'train.csv').fillna("this is nonetype")
    df_test = pd.read_csv(input_folder + 'test.csv').fillna("this is nonetype")
    # ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    print(list(df_train))
    return df_train, df_test


def get_list_of_questions(df_train):
    q1 = df_train['question1'].tolist()
    q2 = df_train['question2'].tolist()
    questions = q1 + q2
    return questions


def read_corpus(questions):
    for i, question in enumerate(questions):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(question), [i])


def train_model(train_corpus):
    skip_window = 4
    embedding_size = EMBEDDING_SIZE  # Dimension of the embedding vector.
    model = Doc2Vec(dm=1, dm_concat=0, size=embedding_size, window=skip_window,
                    negative=20, hs=0, min_count=5, workers=cpus, iter=30)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count)
    return model


print('Number of cpus: %s' % cpus)
print('Reading data...')
df_train, df_test = read_data()
questions_train = get_list_of_questions(df_train)
questions_test = get_list_of_questions(df_test)
questions = questions_train + questions_test

if os.path.isfile(model_folder):
    print('Model exists! Loading the model...')
    model = gensim.models.Word2Vec.load(model_folder)
else:
    train_corpus = list(read_corpus(questions))
    model = train_model(train_corpus)
    model.save(model_folder)

len_docs = len(model.docvecs)
rand_doc = np.random.randint(len_docs)

closest_doc2 = model.docvecs.most_similar([model.docvecs[rand_doc]], topn=4)
for index, sim in closest_doc2:
    print(questions[index])
    print(sim)


print('Finding similar questions to:')
print(df_test['question1'][2])
v1 = model.infer_vector(gensim.utils.simple_preprocess(questions[closest_doc2[2][0]]))
closest_doc2 = model.docvecs.most_similar([v1], topn=4)
for index, sim in closest_doc2:
    print(questions[index])
    print(sim)


norm_vec = np.array([vec for vec in model.docvecs])
norm_vec = norm_vec / np.sqrt(np.sum(np.square(norm_vec), axis=1, keepdims=True))

print(norm_vec[rand_doc].dot(norm_vec[closest_doc2[1][0]]))

q1 = df_train['question1'].tolist()
q2 = df_train['question2'].tolist()

tmp = []
for i, question in enumerate(q1):
    v1 = model.infer_vector(question)
    tmp.append(v1)

questions1 = np.array(tmp)

tmp = []
for i, question in enumerate(q2):
    v2 = model.infer_vector(question)
    tmp.append(v2)

questions2 = np.array(tmp)

labels = df_train['is_duplicate']


inputs1 = Input(shape=(EMBEDDING_SIZE,))
inputs2 = Input(shape=(EMBEDDING_SIZE,))
x1 = Dense(64, activation='linear')(inputs1)
x2 = Dense(64, activation='linear')(inputs2)
inputs = concatenate([x1, x2])
x = Dense(1, activation='softmax')(inputs)
predictions = Dense(1, activation='sigmoid')(x)
nn = Model(inputs=[inputs1, inputs2], outputs=predictions)
nn.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

nn.fit([questions1, questions2], labels, epochs=17, validation_split=0.3)  # starts training

q1 = df_test['question1'].tolist()
q2 = df_test['question2'].tolist()
ids = df_test['test_id']

tmp = []
for i, question in enumerate(q1):
    v1 = model.infer_vector(question)
    tmp.append(v1)
questions1 = np.array(tmp)

tmp = []
for i, question in enumerate(q2):
    v2 = model.infer_vector(question)
    tmp.append(v2)
questions2 = np.array(tmp)

results = nn.predict([questions1, questions2])
submission = pd.DataFrame({'test_id': ids, 'is_duplicate': results.ravel()})
submission.to_csv('doc2vec_deep_nn.csv', index=False)