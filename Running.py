import os

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from Model.LinearModel import LinearModel
from Model.CNNModel import CNNModel
from Preprocess.PrepareData import getIdxData, getTFData


def train(model, train_text, train_numerical, train_labels, num_epochs):
    # train the model for 1 epoch
    batch_size = model.batch_size
    num_samples = len(train_text)
    epoch_loss_list = []
    epoch_accuracy_list = []
    for t in range(num_epochs):
        loss_list = []
        accuracy_list = []
        sum_loss = 0.0
        for idx in range(0, num_samples, batch_size):
            batch_text = train_text[idx: min(idx + batch_size, num_samples)]
            batch_numerical = train_numerical[idx: min(idx + batch_size, num_samples)]
            batch_labels = train_labels[idx: min(idx + batch_size, num_samples)]
            with tf.GradientTape() as tape:
                pred_Y = model(batch_text, batch_numerical)
                loss = model.loss(pred_Y, batch_labels)
                acc = model.accuracy(pred_Y, batch_labels)
                sum_loss += loss
            loss_list.append(loss)
            accuracy_list.append(acc)
            # Update the model after every batch
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('Epoch %d\tLoss: %.3f | Acc: %.3f' % (t, np.mean(loss_list), np.mean(accuracy_list)))
        epoch_loss_list.append(loss_list)
        epoch_accuracy_list.append(accuracy_list)
    return model, epoch_loss_list, epoch_accuracy_list


def evaluate(model, test_text, test_numerical, test_labels):
    # test the model
    pred_Y = model(test_text, test_numerical)
    loss = model.loss(pred_Y, test_labels)
    acc = model.accuracy(pred_Y, test_labels)
    print('Test Data \tLoss: %.3f | Acc: %.3f' % (loss, acc))


def visualizeTrainHistory(epoch_loss_list, epoch_acc_list):
    avg_epoch_loss = [np.mean(each) for each in epoch_loss_list]
    avg_epoch_accuracy = [np.mean(each) for each in epoch_acc_list]

    plt.subplot(1, 2, 1)
    plt.title("Loss Value")
    plt.plot(avg_epoch_loss, lw=4)
    plt.xticks(np.arange(0, len(avg_epoch_loss)+1, 5), np.arange(0, len(avg_epoch_loss)+1, 5)+1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.title("Classification Accuracy")
    plt.plot(avg_epoch_accuracy, lw=4)
    plt.xticks(np.arange(0, len(avg_epoch_accuracy)+1, 5), np.arange(0, len(avg_epoch_accuracy)+1, 5)+1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def main(data_filename, model_name, num_epochs=5, visualize=False):
    print("=" * 70)
    print("Running preprocessing...")
    if model_name == "linear":
        train_text, test_text, train_numerical, test_numerical, train_labels, test_labels, vocab_dict = getTFData(data_filename)
    elif model_name == "CNN":
        train_text, test_text, train_numerical, test_numerical, train_labels, test_labels, vocab_dict = getIdxData(data_filename)
    else:
        raise NotImplemented("Not implemented model {}.".format(model_name))
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_labels.shape[1]
    print("Label size = {}".format(label_size))
    print("Preprocessing complete.")
    # -----------------------------
    print("=" * 70)
    if model_name == "linear":
        model = LinearModel(vocab_size, label_size)
    elif model_name == "CNN":
        model = CNNModel(vocab_size, label_size)
    else:
        raise NotImplemented("The model {} is not implemented!".format(model_name))
    print("Start training...")
    model, epoch_loss_list, epoch_accuracy_list = train(model, train_text, train_numerical, train_labels, num_epochs)
    # save final model
    checkpoint_dir = "{}".format(model_name)
    if checkpoint_dir not in os.listdir("./checkpoint"):
        os.mkdir("./checkpoint/{}".format(checkpoint_dir))
    model.save("./checkpoint/{}".format(checkpoint_dir))
    if visualize:
        visualizeTrainHistory(epoch_loss_list, epoch_accuracy_list)
    print("=" * 70)
    print("Start testing...")
    evaluate(model, test_text, test_numerical, test_labels)


if __name__ == '__main__':
    data_filename = "./data/top5_label_data_new.csv"
    main(data_filename, model_name="linear", num_epochs=10, visualize=False)
    # main(data_filename, model_name="CNN", num_epochs=5, visualize=False)