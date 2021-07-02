# -*- coding:utf-8 -*-
"""
 @Time: 2021/6/30 下午3:21
 @Author: LiuHe
 @File: main.py
 @Describe: 测试TextCNN
"""
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from task2.TextCNN import TextCNN

max_features = 5000
max_len = 400
batch_size = 32
embedding_dims = 50
epochs = 10


model = TextCNN(max_len, max_features, embedding_dims)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy'])
print(model.summary())