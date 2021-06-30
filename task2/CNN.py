# -*- coding:utf-8 -*-
"""
 @Time: 2021/6/29 下午1:52
 @Author: LiuHe
 @File: CNN.py
 @Describe: 使用CNN进行文本分类
"""
import os
import time
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import sklearn
import random
import matplotlib.pyplot as plt


dir_all_data = 'data/task2_all_data.tsv'

BATCH_SIZE = 10
data_all = pd.read_csv(dir_all_data, sep='\t')

idx = np.arange(data_all.shape[0])
seed = 0

np.random.seed(seed)
np.random.shuffle(idx)

# 将数据集划分为训练集、验证集、测试集，并进行保存
train_size = int(len(idx) * 0.6)
test_size = int(len(idx) * 0.8)

train_data = data_all[:train_size]
test_data = data_all[train_size: test_size]
dev_data = data_all[test_size:]


# train_data.to_csv('data/task2_train_tf.csv', index=False)
# test_data.to_csv('data/task2_test_tf.csv', index=False)
# dev_data.to_csv('data/task2_dev_tf.csv', index=False)
data_all.iloc[idx[:train_size], :].tocsv('data/task2_train_tf.csv', index=False)
data_all.iloc[idx[train_size: test_size], :].tocsv('data/task2_test_tf.csv', index=False)
data_all.iloc[idx[test_size:], :].tocsv('data/task2_dev_tf.csv', index=False)

train_data = pd.read_csv("data/task2_train_tf.csv")
test_data = pd.read_csv("data/task2_test_tf.csv")

train_data_sentences = train_data['Phrase']
train_data_labels = train_data['Sentiment']

test_data_sentences = test_data['Phrase']
test_data_labels = test_data['Sentiment']

dev_data_sentences = dev_data['Phrase']
dev_data_labels = dev_data['Sentiment']

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


# model = keras.models.Sequential([
#     keras.layers.Embedding(vocab_size, embedding_dim),
#     keras.layers.Conv1D(256, 5, activation='relu', padding='same'),
#     keras.layers.MaxPool1D(3, 3, padding='same'),
#     keras.layers.Conv1D(256, 4, activation='relu', padding="same"),
#     keras.layers.MaxPool1D(pool_size=47),
#     keras.layers.Conv1D(256, 5, activation="relu", padding="same"),
#     keras.layers.MaxPool1D(pool_size=46),
#     keras.layers.Flatten(),
#     keras.layers.Dropout(dropout_p),
#     keras.layers.Dense(6, activation='softmax')
# ])

main_input = keras.layers.Input(shape=(max_length,))
embed = keras.layers.Embedding(vocab_size, embedding_dim,
                               input_length=max_length)(main_input)

cnn1 = keras.layers.Conv1D(256, 3, padding='same', activation='relu',
                           strides=1)(embed)
cnn1 = keras.layers.MaxPool1D(pool_size=48)(cnn1)

cnn2 = keras.layers.Conv1D(256, 4, padding='same', activation='relu',
                           strides=1)(embed)
cnn2 = keras.layers.MaxPool1D(pool_size=48)(cnn2)

cnn3 = keras.layers.Conv1D(256, 5, padding='same', activation='relu',
                           strides=1)(embed)
cnn3 = keras.layers.MaxPool1D(pool_size=48)(cnn3)

cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = keras.layers.Flatten()(cnn)
drop = keras.layers.Dropout(dropout_p)(flat)
main_output = keras.layers.Dense(5, activation='softmax')(drop)

model = keras.Model(inputs=main_input, outputs=main_output)
# print(model.summary())

# model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])

print(model.summary())

train_data_labels = keras.utils.to_categorical(train_data_labels)
test_data_labels = keras.utils.to_categorical(test_data_labels)
print(train_data_labels[:5])


model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(), metrics=[
        'accuracy'])
history = model.fit(train_sequence_padded, train_data_labels, epochs=30,
                    batch_size=32, validation_data=(test_sequence_padded,
                                                    test_data_labels),
                    verbose=1)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('epoches')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

