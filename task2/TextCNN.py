# -*- coding:utf-8 -*-
"""
 @Time: 2021/6/30 下午2:30
 @Author: LiuHe
 @File: TextCNN.py
 @Describe: 复现TextCNN
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, \
    GlobalAveragePooling1D, concatenate, Dropout


class TextCNN(Model):

    def __init__(self,
                 max_len,
                 max_features,
                 embedding_dims,
                 kernel_size=[3, 4, 5],
                 class_num=1,
                 last_activation='sigmoid'):
        """

        :param max_len:
        :param max_features:
        :param embedding_dims:
        :param kernel_size:
        :param class_num:
        :param last_activation:
        """
        super(TextCNN, self).__init__()
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_size
        self.class_num = class_num
        self.last_activation = last_activation

        self.embedding = Embedding(self.max_features, self.embedding_dims,
                                   input_length=self.max_len)
        self.convs = []
        self.max_poolings = []

        for kernel_size in self.kernel_sizes:
            self.convs.append(
                Conv1D(128, kernel_size, activation='relu')
            )
            self.max_poolings.append(GlobalAveragePooling1D())
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('the rank of inputs of TextCNN must be 2, '
                             'but now is %d'%len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.max_len:
            raise ValueError('The max length of TextCNN must be %d'%self.max_len)

        embedding = self.embedding(inputs)
        convs = []
        for i in range(len((self.kernel_sizes))):
            c = self.convs[i](embedding)
            c = self.max_poolings[i](c)
            convs.append(c)
        x = concatenate()(convs)
        output = self.classifier(x)
        return output