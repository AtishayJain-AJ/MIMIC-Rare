'''
Description:
    The classification model with only linear layers.

Author:
    Jiaqi Zhang
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


class LinearModel(tf.keras.Model):

    def __init__(self, vocab_size, label_size):
        super(LinearModel, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.text_latent_size = 64
        self.numerical_latent_size = 64
        self.batch_size = 32
        self.label_size = label_size # the number of unique ICD9 codes

        # Model architecture
        self.text_module = Sequential([
            Dense(self.text_latent_size, name="text_linear", activation="relu"),
            # Dropout(0.1)
        ])
        self.numerical_module = Sequential([
            Dense(self.numerical_latent_size, name="numerical_linear", activation="relu"),
            # Dropout(0.1)
        ])
        self.prediction_head = Sequential([
            Dense(256, activation="relu"),
            # Dropout(0.3),
            Dense(self.label_size, activation="softmax")
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


    def call(self, input_text, input_value):
        text_latent = self.text_module(input_text)
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
    from Preprocess.PrepareData import get_data
    train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = get_data(
        "../data/binary_label_data.csv")
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_label.shape[1]
    print("Label size = {}".format(label_size))
    # ---------------------------------------
    batch_idx = np.arange(32)
    model = LinearModel(vocab_size, label_size)
    probs = model.call(train_text[batch_idx], train_numerical[batch_idx])
    loss = model.loss(probs, train_label[batch_idx])
    print("Loss value = {}".format(loss))
    acc = model.accuracy(probs, train_label[batch_idx])
    print("Accuracy value = {}".format(acc))