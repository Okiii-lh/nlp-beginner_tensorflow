# -*- coding:utf-8 -*-
"""
 @Time: 2021/6/30 下午4:11
 @Author: LiuHe
 @File: text_cnn_test.py
 @Describe:
"""
from task2.text_cnn import TextCNN

max_features = 5000
max_len = 400
batch_size = 32
embedding_dims = 50
epochs = 10


model = TextCNN(max_len, max_features, embedding_dims)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
    'accuracy'])
print(model.summary())