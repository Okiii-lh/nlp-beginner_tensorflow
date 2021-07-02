# -*- coding:utf-8 -*-
"""
 @Time: 2021/7/2 下午4:06
 @Author: LiuHe
 @File: RNN.py
 @Describe: TextRNN
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTM


class TextRNN(Model):
    def __init__(self,
                 max_len,
                 max_features,
                 embedding_dim,
                 class_num,
                 last_activation='sigmoid'):
        """
        TextRNN 初始化
        :param max_len: 句长
        :param max_features: 字典长度
        :param embedding_dim: 每个词的向量维度
        :param class_num: 类别数量
        :param last_activation: 全连接层激活函数
        """
        super(TextRNN, self).__init__()
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_dims = embedding_dim
        self.class_num = class_num
        self.last_activation = last_activation

        self.embedding = Embedding(self.max_features, self.embedding_dims,
                                   self.max_len)

        self.rnn = LSTM(128)
        self.classifier = Dense(self.class_num, activation=last_activation)

    def call(self, inputs):
        embedding = self.embedding(inputs)
        x = self.rnn(embedding)
        output = self.classifier(x)
        return output
