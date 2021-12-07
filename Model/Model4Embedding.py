'''
Description:
    The classification model for uni-modal features.

Author:
    Jiaqi Zhang
'''
import os

import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Embedding, Flatten
from tensorflow_addons.layers import NoisyDense


class NumericalModel(tf.keras.Model):

    def __init__(self, label_size):
        super(NumericalModel, self).__init__()
        # Parameters
        self.numerical_latent_size = 16
        self.batch_size = 16
        self.label_size = label_size # the number of unique ICD9 codes

        self.numerical_module = Sequential([
            # Dense(256, name="numerical_linear1", activation=tf.nn.relu),
            # Dropout(0.8),
            Dense(self.numerical_latent_size, name="numerical_linear2", activation=tf.nn.relu),
            Dropout(0.8)
        ])
        self.prediction_head = Sequential([
            # Dense(16, activation="relu"),
            # Dropout(0.8),
            Dense(self.label_size, activation="softmax"),
            Dropout(0.8),
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, placehold, input_value):
        numerical_latent = self.numerical_module(input_value)
        prediction = self.prediction_head(numerical_latent)
        # prediction = self.prediction_head(input_value)
        return prediction

    def loss(self, probs, labels):
        labels_code = tf.argmax(labels, axis=1)
        loss = self.loss_func(labels_code, probs)
        return loss

    def accuracy(self, probs, labels):
        pred_labels = tf.argmax(probs, axis=1)
        true_labels = tf.argmax(labels, axis=1)
        return tf.reduce_mean(tf.cast(tf.math.equal(pred_labels, true_labels), tf.float32))


class TextLinearModel(tf.keras.Model):

    def __init__(self, vocab_size, label_size):
        super(TextLinearModel, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.text_latent_size = 256
        self.batch_size = 16
        self.label_size = label_size # the number of unique ICD9 codes

        # Model architecture
        self.text_module = Sequential([
            Dense(self.text_latent_size, name="text_linear", activation=tf.nn.relu,
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dropout(0.8)
        ])
        self.prediction_head = Sequential([
            # Dense(16, activation="relu"),
            # Dropout(0.8),
            Dense(self.label_size, activation="softmax",
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    def call(self, input_text, placehold):
        text_latent = self.text_module(input_text)
        prediction = self.prediction_head(text_latent)
        return prediction


    def loss(self, probs, labels):
        labels_code = tf.argmax(labels, axis=1)
        loss = self.loss_func(labels_code, probs)
        return loss

    def accuracy(self, probs, labels):
        pred_labels = tf.argmax(probs, axis=1)
        true_labels = tf.argmax(labels, axis=1)
        return tf.reduce_mean(tf.cast(tf.math.equal(pred_labels, true_labels), tf.float32))


class TextCNNModel(tf.keras.Model):

    def __init__(self, vocab_size, label_size):
        super(TextCNNModel, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.text_latent_size = 64
        self.batch_size = 32
        self.label_size = label_size # the number of unique ICD9 codes

        # Model architecture
        self.text_module = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, name="text_embedding"),
            Conv1D(16, 3, name="text_conv", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            MaxPool1D(2, name="text_pool"),

            Conv1D(16, 3, name="text_conv2", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            MaxPool1D(2, name="text_pool2"),
        ])
        self.text_dense = Sequential([
            Flatten(name="text_flatten"),
            Dense(self.text_latent_size, name="text_linear", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dropout(0.8)
        ])
        self.prediction_head = Sequential([
            # Dense(256, activation="relu", name="prediction_linear1",
            #       kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            # Dropout(0.8),
            Dense(self.label_size, activation="softmax", name="prediction_linear2",
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    def call(self, input_text, placehold):
        text_latent = self.text_module(input_text)
        text_latent = self.text_dense(text_latent)
        prediction = self.prediction_head(text_latent)
        return prediction


    def loss(self, probs, labels):
        labels_code = tf.argmax(labels, axis=1)
        loss = self.loss_func(labels_code, probs)
        return loss

    def accuracy(self, probs, labels):
        pred_labels = tf.argmax(probs, axis=1)
        true_labels = tf.argmax(labels, axis=1)
        return tf.reduce_mean(tf.cast(tf.math.equal(pred_labels, true_labels), tf.float32))


if __name__ == '__main__':
    from Preprocess.PrepareData import getIdxData, getTFData
    train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = getTFData(
        "../data/top10_label_data_new.csv")
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_label.shape[1]
    print("Label size = {}".format(label_size))
    batch_idx = np.arange(32)
    # ---------------------------------------
    model = NumericalModel(label_size)
    probs = model.call(None, train_numerical[batch_idx])
    loss = model.loss(probs, train_label[batch_idx])
    print("[ Numerical Model ] Loss value = {}".format(loss))
    acc = model.accuracy(probs, train_label[batch_idx])
    print("[ Numerical Model ] Accuracy value = {}".format(acc))
    # ---------------------------------------
    model = TextLinearModel(vocab_size, label_size)
    probs = model.call(train_text[batch_idx], None)
    loss = model.loss(probs, train_label[batch_idx])
    print("[ Text Linear Model ] Loss value = {}".format(loss))
    acc = model.accuracy(probs, train_label[batch_idx])
    print("[ Text Linear Model ] Accuracy value = {}".format(acc))
    # ---------------------------------------
    train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = getIdxData(
        "../data/top10_label_data_new.csv")
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_label.shape[1]
    print("Label size = {}".format(label_size))
    batch_idx = np.arange(32)
    model = TextCNNModel(vocab_size, label_size)
    probs = model.call(train_text[batch_idx], None)
    loss = model.loss(probs, train_label[batch_idx])
    print("[ Text CNN Model ] Loss value = {}".format(loss))
    acc = model.accuracy(probs, train_label[batch_idx])
    print("[ Text CNN Model ] Accuracy value = {}".format(acc))