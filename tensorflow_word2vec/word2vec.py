from __future__ import print_function

import collections
import math
import random
from subprocess import check_output
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves import range
from tensorflow.contrib.tensorboard.plugins import projector

print(check_output(["ls", "../data"]).decode("utf8"))
LOG_DIR = './graphs'


def read_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv').fillna("this is nonetype")
    print('Data size %d' % len(df_train))
    print('Data headers %s' % df_train.columns.values)
    return df_train, df_test


df_train, df_test = read_data()


def extract_words(df_train):
    words = list()
    for index, row in df_train.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        if not q1 or not q2 or not isinstance(q1, basestring) \
                or not isinstance(q2, basestring):
            continue
        q_words = q1.split()
        for word in q_words:
            words.append(word)
        q_words = q2.split()
        for word in q_words:
            words.append(word)
        words.append(" ")
    return words


vocabulary_size = 50000  # This has to be less than size of count
words = extract_words(df_train)
print('Number of words: %d' % len(words))

metadata = os.path.join(LOG_DIR, 'metadata.tsv')
with open(metadata, 'w') as metadata_file:
    for row in words:
        metadata_file.write('%s\n' % row)


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

q1 = df_test['question1'][0]
tmp = []
tmp2 = []

for i, word in enumerate(q1.split()):
    if word in dictionary:
        key = dictionary[word]
        tmp.append(key)
q1_words = np.array(tmp)
test_size = len(q1_words)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        test_dataset = tf.constant(q1_words, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        with tf.name_scope('lookupembeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
        tf.summary.scalar('loss', loss)

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    test_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, test_dataset)
    similarity_test = tf.matmul(test_embeddings, tf.transpose(normalized_embeddings))
    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(LOG_DIR, session.graph)

    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embeddings.name
    embedding_config.metadata_path = metadata
    projector.visualize_embeddings(writer, config)

    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        s, _, loss_val = session.run([merged_summary, optimizer, loss], feed_dict=feed_dict)
        writer.add_summary(s, step)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            saver.save(session, os.path.join(LOG_DIR, LOG_DIR + "model.ckpt"), step)
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
        if step % 50000 == 0:
            print('-----------------Testing--------------------------')
            sim = similarity_test.eval()
            for i in range(test_size):
                test_word = reverse_dictionary[q1_words[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % test_word
                for k in range(top_k):
                    a = nearest[k]
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()
    saver.save(session, 'word2vec_model', global_step=step)
    writer.close()
