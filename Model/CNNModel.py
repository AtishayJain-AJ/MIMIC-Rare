'''
Description:
    The classification model with 1d convolutional layers.

Author:
    Jiaqi Zhang
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Embedding, Flatten


# ==========================================
#          !!!!! WARNING !!!!!
#
#      This model is not completed.
# ==========================================

class CNNModel(tf.keras.Model):

    def __init__(self, vocab_size, label_size):
        super(CNNModel, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.text_latent_size = 64
        self.numerical_latent_size = 64
        self.batch_size = 32
        self.label_size = label_size # the number of unique ICD9 codes

        # Model architecture
        self.text_module = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, name="text_embedding"),
            Conv1D(16, 3, name="text_conv", activation="relu"),
            MaxPool1D(2, name="text_pool"),
        ])
        self.text_dense = Sequential([
            Flatten(name="text_flatten"),
            Dense(self.text_latent_size, name="text_linear", activation="relu"),
            Dropout(0.25)
        ])
        self.numerical_module = Sequential([
            Dense(self.numerical_latent_size, name="numerical_linear", activation="relu"),
            Dropout(0.25)
        ])
        self.prediction_head = Sequential([
            Dense(256, activation="relu", name="prediction_linear1"),
            Dropout(0.25),
            Dense(self.label_size, activation="softmax", name="prediction_linear2")
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    def call(self, input_text, input_value):
        text_latent = self.text_module(input_text)
        text_latent = self.text_dense(text_latent)
        numerical_latent = self.numerical_module(input_value)
        concat_latent = tf.concat([text_latent, numerical_latent], axis=1)
        prediction = self.prediction_head(concat_latent)
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
    from Preprocess.PrepareData import getIdxData
    train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = getIdxData(
        "../data/binary_label_data.csv")
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_label.shape[1]
    print("Label size = {}".format(label_size))
    # ---------------------------------------
    batch_idx = np.arange(32)
    model = CNNModel(vocab_size, label_size)
    probs = model.call(train_text[batch_idx], train_numerical[batch_idx])
    loss = model.loss(probs, train_label[batch_idx])
    print("Loss value = {}".format(loss))
    acc = model.accuracy(probs, train_label[batch_idx])
    print("Accuracy value = {}".format(acc))