import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

import seaborn as sbn
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Antique_5
params = {
    "legend.fontsize": 15,
    "legend.frameon": False,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
    # "figure.dpi": 600,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
}
plt.rcParams.update(params)


from Model.LinearModel import LinearModel
from Model.CNNModel import CNNModel
from Model.UnimodalModel import NumericalModel, TextLinearModel, TextCNNModel
from Preprocess.PrepareData import getIdxData, getTFData, getAllData


def train(model, train_text, train_numerical, train_labels, test_text, test_numerical, test_labels, num_epochs):
    batch_size = model.batch_size
    num_samples = len(train_text)
    epoch_loss_list = []
    epoch_accuracy_list = []
    evaluate_loss_list = []
    evaluate_accuracy_list = []
    for t in range(num_epochs):
        print("#" * 50)
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
        evaluate_loss, evaluate_acc = evaluate(model, test_text, test_numerical, test_labels)
        evaluate_loss_list.append(evaluate_loss)
        evaluate_accuracy_list.append(evaluate_acc)
    return model, epoch_loss_list, epoch_accuracy_list, evaluate_loss_list, evaluate_accuracy_list


def evaluate(model, test_text, test_numerical, test_labels):
    # test the model
    pred_Y = model(test_text, test_numerical)
    loss = model.loss(pred_Y, test_labels)
    acc = model.accuracy(pred_Y, test_labels)
    print('Test Data \tLoss: %.3f | Acc: %.3f' % (loss, acc))
    return float(loss), float(acc)


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


def main(data_filename, num_epochs=5):
    print("=" * 70)
    print("Running preprocessing...")
    train_text, test_text, train_text_id, test_text_id, train_numerical, test_numerical, \
    train_labels, test_labels, vocab_dict = getAllData(data_filename)
    vocab_size = len(vocab_dict.keys())
    print("Vocabulary size = {}".format(vocab_size))
    label_size = train_labels.shape[1]
    print("Label size = {}".format(label_size))
    print("Preprocessing complete.")
    # -----------------------------
    print("=" * 70)
    print("Constructing model...")
    linear_model = LinearModel(vocab_size, label_size)
    cnn_model = CNNModel(vocab_size, label_size)
    numerical_model = NumericalModel(label_size)
    text_linear_model = TextLinearModel(vocab_size, label_size)
    text_cnn_model = TextCNNModel(vocab_size, label_size)

    print("Start training...")
    trial_res = {}
    name_model = {
        "linear":linear_model, "cnn":cnn_model, "numerical":numerical_model,
        "text_linear":text_linear_model, "text_cnn": text_cnn_model
    }
    for n in name_model:
        print("-" * 70)
        print("[ Model {} ]".format(n))
        if "cnn" in n:
            tmp_train_text = train_text_id
            tmp_test_text = test_text_id
        else:
            tmp_train_text = train_text
            tmp_test_text = test_text
        model, epoch_loss_list, epoch_accuracy_list, evaluate_loss_list, evaluate_accuracy_list = \
            train(name_model[n],
                  tmp_train_text, train_numerical, train_labels,
                  tmp_test_text, test_numerical, test_labels,
                  num_epochs)
        trial_res[n] = {
            "train_epoch_loss": [np.mean(each) for each in epoch_loss_list],
            "train_epoch_acc": [np.mean(each) for each in epoch_accuracy_list],
            "evaluate_epoch_loss": evaluate_loss_list,
            "evaluate_epoch_acc": evaluate_accuracy_list,
        }
    return trial_res

# -------------------------------------------------------

def visualize(res, label_size):
    # re-organize data
    model_res = {
        "linear":[], "cnn":[], "numerical":[],
        "text_linear":[], "text_cnn": []
    }
    for t in range(len(res)):
        for m in model_res:
            model_res[m].append(res[t][m])
    print(model_res["linear"][0].keys())
    # Evaluate accuracy of each model
    model_evaluate_acc = {
        m: [np.max(trial["evaluate_epoch_acc"]) for trial in model_res[m]]
        for m in model_res
    }
    model_evaluate_acc = {
        m: (np.mean(model_evaluate_acc[m]), np.std(model_evaluate_acc[m]))
        for m in model_evaluate_acc
    }
    print("Evaluate accuracy:")
    print(model_evaluate_acc)
    # ------------------------------------------
    # # Train and validate loss and accuracy
    _plotLossAndAcc(model_res, "linear", label_size)
    _plotLossAndAcc(model_res, "cnn", label_size)
    # ------------------------------------------


def _plotLossAndAcc(model_res, model_name, label_size):
    linear_train_loss = np.asarray([trial["train_epoch_loss"] for trial in model_res[model_name]])
    linear_evaluate_loss = np.asarray([trial["evaluate_epoch_loss"] for trial in model_res[model_name]])
    avg_train_loss = np.mean(linear_train_loss, axis=0)
    std_train_loss = np.std(linear_train_loss, axis=0)
    avg_evaluate_loss = np.mean(linear_evaluate_loss, axis=0)
    std_evaluate_loss = np.std(linear_evaluate_loss, axis=0)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(np.arange(10), avg_train_loss, "o-", ms=10, lw=3.5, label="Train", color=Antique_5.mpl_colors[0])
    plt.fill_between(
        np.arange(10),
        avg_train_loss - std_train_loss,
        avg_train_loss + std_train_loss,
        color=Antique_5.mpl_colors[0],
        alpha=0.4,
        linewidth=0.0)
    plt.plot(np.arange(10), avg_evaluate_loss, "o-", ms=10, lw=3.5, label="Evaluate", color=Antique_5.mpl_colors[1])
    plt.fill_between(
        np.arange(10),
        avg_evaluate_loss - std_evaluate_loss,
        avg_evaluate_loss + std_evaluate_loss,
        color=Antique_5.mpl_colors[1],
        alpha=0.4,
        linewidth=0.0)
    plt.xticks(np.arange(10), np.arange(1, 11))
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Cross-Entropy Loss", fontsize=15)
    plt.legend(frameon=False, fontsize=15)
    plt.tight_layout()
    plt.savefig("./figs/top{}-{}-loss.pdf".format(label_size, model_name))
    plt.show()
    plt.close()
    # Train and validate loss and accuracy
    linear_train_acc = np.asarray([trial["train_epoch_acc"] for trial in model_res[model_name]])
    linear_evaluate_acc = np.asarray([trial["evaluate_epoch_acc"] for trial in model_res[model_name]])
    avg_train_acc = np.mean(linear_train_acc, axis=0)
    std_train_acc = np.std(linear_train_acc, axis=0)
    avg_evaluate_acc = np.mean(linear_evaluate_acc, axis=0)
    std_evaluate_acc = np.std(linear_evaluate_acc, axis=0)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(np.arange(10), avg_train_acc, "o-", ms=10, lw=3.5, label="Train", color=Antique_5.mpl_colors[0])
    plt.fill_between(
        np.arange(10),
        avg_train_acc - std_train_acc,
        avg_train_acc + std_train_acc,
        color=Antique_5.mpl_colors[0],
        alpha=0.4,
        linewidth=0.0
    )
    plt.plot(np.arange(10), avg_evaluate_acc, "o-", ms=10, lw=3.5, label="Evaluate", color=Antique_5.mpl_colors[1])
    plt.fill_between(
        np.arange(10),
        avg_evaluate_acc - std_evaluate_acc,
        avg_evaluate_acc + std_evaluate_acc,
        color=Antique_5.mpl_colors[1],
        alpha=0.4,
        linewidth=0.0
    )
    plt.xticks(np.arange(10), np.arange(1, 11))
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(frameon=False, fontsize=15)
    plt.tight_layout()
    plt.savefig("./figs/top{}-{}-accuracy.pdf".format(label_size, model_name))
    plt.show()
    plt.close()


def _OGR(model_res, model_name):
    train_loss = np.asarray([trial["train_epoch_loss"] for trial in model_res[model_name]])
    avg_train_loss = np.mean(train_loss, axis=0)
    evaluate_loss = np.asarray([trial["evaluate_epoch_loss"] for trial in model_res[model_name]])
    avg_evaluate_loss = np.mean(evaluate_loss, axis=0)
    # -----
    delta_G = np.asarray([avg_evaluate_loss[0]-each for each in avg_evaluate_loss]) # difference of evaluate loss
    diff = np.asarray([
        avg_evaluate_loss[idx]-avg_train_loss[idx] for idx in np.arange(len(avg_train_loss))
    ]) # difference between train and evaluate loss
    delta_O = np.asarray([diff[0]-each for each in diff])
    OGR_per_epoch = delta_O / delta_G
    return diff / avg_train_loss


def _plotLossDiff(model_diff, label_size):
    for each in ["linear", "cnn"]:
        # tmp = model_diff[each]
        # tmp /= np.max(tmp)
        plt.plot(model_diff[each],  label=each)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # for label_size in [2, 3, 4, 5, 10]:
        # label_size = 3
        # --------------------------------------------------------------------
        # data_filename = "./data/top{}_label_data_new.csv".format(label_size)
        # res = []
        # for _ in range(5):
        #     trial_res = main(data_filename, num_epochs=10)
        #     res.append(trial_res)
        # np.save("./top{}_label-modality_comparison.npy".format(label_size), res)
    # --------------------------------------------------------------------
    for label_size in [2, 3, 4, 5, 10]:
        print("-" * 70)
        print("[ {} Codes ]".format(label_size))
        res = np.load("./top{}_label-modality_comparison.npy".format(label_size), allow_pickle=True)
        visualize(res, label_size)