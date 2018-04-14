#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf

from keras_utils.utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')

tf.app.flags.DEFINE_string('train_file_path', '../data/data_1_train.csv', 'training file')
# tf.app.flags.DEFINE_string('validate_file_path', 'data/twitter/validate.raw', 'validating file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/test.raw', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', '../embeddings/glove.6B.300d.txt',
                           'embedding file')
# tf.app.flags.DEFINE_string('word_id_file_path', 'data/twitter/word_id.txt',
# 'word-id mapping file') # don't need this, already being calculated
tf.app.flags.DEFINE_string('type', 'TD', 'model type: ''(default), TD or TC')


class LSTM(object):
    def __init__(self, embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.01,
                 n_class=3, max_sentence_len=50, l2_reg=0., display_step=4, n_iter=100, type_=''):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.display_step = display_step
        self.n_iter = n_iter
        self.type_ = type_
        self.datasets = load_and_clean()
        self.glove = load_embedding_matrix(self.datasets[0])
        self.word_embedding = tf.constant(self.glove, name='word_embedding')