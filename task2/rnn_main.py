# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/2 下午4:29
 @Author: LiuHe
 @File: rnn_main.py
 @Describe:
"""
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from task2.text_rnn import TextRNN


dir_all_data = 'data/task2_all_data.tsv'

BATCH_SIZE = 10
data_all = pd.read_csv(dir_all_data, sep='\t')

idx = np.arange(data_all.shape[0])
seed = 0
np.random.seed(seed)
np.random.shuffle(idx)

train_size = int(len(idx) * 0.6)
test_size = int(len(idx) * 0.8)

data_all.iloc[idx[:train_size], :].to_csv('data/task2_train_tf.csv',
                                          index=False)
data_all.iloc[idx[train_size:test_size], :].to_csv("data/task2_test_tf.csv",
                                                   index=False)
data_all.iloc[idx[test_size:], :].to_csv("data/task2_dev_tf.csv", index=False)

train_data = pd.read_csv("data/task2_train_tf.csv")
test_data = pd.read_csv("data/task2_test_tf.csv")

train_data_sentences = train_data['Phrase']
train_data_labels = train_data['Sentiment']

test_data_sentences = test_data['Phrase']
test_data_labels = test_data['Sentiment']

print(train_data_sentences[:5])
print(train_data_labels[:5])
print(test_data_sentences[:5])
print(test_data_labels[:5])

vocab_size = 16473
embedding_dim = 50
dropout_p = 0.5
max_length = 50
pad_token = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=pad_token)
tokenizer.fit_on_texts(train_data_sentences)
train_sequence = tokenizer.texts_to_sequences(train_data_sentences)
train_sequence_padded = pad_sequences(train_sequence, maxlen=max_length)

test_sequence = tokenizer.texts_to_sequences(test_data_sentences)
test_sequence_padded = pad_sequences(test_sequence, maxlen=max_length)
print("train_size: ", train_sequence_padded.shape)
print("test_size: ", test_sequence_padded.shape)


model = TextRNN(max_length, embedding_dim, vocab_size)

train_data_labels = keras.utils.to_categorical(train_data_labels)
test_data_labels = keras.utils.to_categorical(test_data_labels)
print(train_data_labels[:5])


model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(), metrics=[
        'accuracy'])
history = model.fit(train_sequence_padded, train_data_labels, epochs=30,
                    batch_size=32, validation_data=(test_sequence_padded,
                                                    test_data_labels))


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('epoches')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')