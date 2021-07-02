# -*- coding:utf-8 -*-
"""
 @Time: 2021/6/30 下午3:41
 @Author: LiuHe
 @File: text_cnn.py
 @Describe: 复现TextCNN
"""
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPool1D, \
    GlobalAveragePooling1D, Concatenate
from tensorflow.keras import Model


class TextCNN(Model):
    def __init__(self,
                 max_len,
                 embedding_dim,
                 max_features,
                 kernel_sizes=[3, 4, 5],
                 class_num=5,
                 last_activation='sigmoid',
                 embedding_matrix=None):
        """
        初始化TextCNN
        :param max_len: 句长度
        :param embedding_dim: 向量长度
        :param max_features: 字典长度
        :param kernel_sizes: 卷积核大小
        :param class_num: 类别数量
        :param last_activation: 全连接层激活函数
        """
        super(TextCNN, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding_matrix = embedding_matrix

        if self.embedding_matrix is None:
            self.embedding = Embedding(self.max_features, self.embedding_dim,
                                        input_length=self.max_len)
        else:
            self.embedding = Embedding(self.max_features, self.embedding_dim,\
                             input_length=self.max_len, weights=[embedding_matrix])
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(
                Conv1D(128, kernel_size, activation='relu')
            )
            self.max_poolings.append(
                GlobalAveragePooling1D()
            )
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        embedding = self.embedding(inputs)

        convs = []
        for i in range(len(self.convs)):
            x = self.convs[i](embedding)
            x = self.max_poolings[i](x)
            convs.append(x)
        x = Concatenate()(convs)
        output = self.classifier(x)
        return output
