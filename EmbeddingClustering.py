import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

import seaborn as sbn
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Bold_3, Bold_5
class_color = Bold_3.mpl_colors
# class_color = Bold_5.mpl_colors

params = {
    "legend.fontsize": 15,
    "legend.frameon": False,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
    "figure.dpi": 600,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
}
plt.rcParams.update(params)


# from Model.LinearModel import LinearModel
# # from Model.CNNModel import CNNModel
from Model.Model4Embedding import NumericalModel, TextLinearModel, TextCNNModel
from Preprocess.PrepareData import getIdxData, getTFData, getAllData

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from SolveOverfitting import train


# -------------------------------------------------------------------------------
label_size = 3
data_filename = "./data/top{}_label_data_new.csv".format(label_size)
print("Run preprocessing...")
train_text, test_text, train_text_id, test_text_id, train_numerical, test_numerical, \
    train_labels, test_labels, vocab_dict = getAllData(data_filename)
vocab_size = len(vocab_dict.keys())
print("Vocabulary size = {}".format(vocab_size))
label_size = train_labels.shape[1]
print("Label size = {}".format(label_size))
print("Preprocessing complete.")

# -------------------------------------------------------------------------------
model = SVC()
model.fit(train_numerical, np.argmax(train_labels, axis=1))
numerical_sc = model.score(test_numerical, np.argmax(test_labels, axis=1))
print("Numerical value classification score = {}".format(numerical_sc))

model = SVC()
model.fit(train_text, np.argmax(train_labels, axis=1))
text_sc = model.score(test_text, np.argmax(test_labels, axis=1))
print("Text IF value classification score = {}".format(text_sc))

# --------------------------------------------------------------------------------
all_numerical = np.concatenate([train_numerical, test_numerical], axis=0)
all_text = np.concatenate([train_text, test_text], axis=0)
all_labels = np.concatenate([np.argmax(train_labels, axis=1), np.argmax(test_labels, axis=1)], axis=0)

pca = PCA(n_components=2)
numerical_embedding = pca.fit_transform(all_numerical)
plt.title("Numerical Values", fontsize=15)
for l in range(label_size):
    tmp = numerical_embedding[np.where(all_labels==l)[0]]
    plt.scatter(tmp[:, 0], tmp[:, 1], label="Class {}".format(l), s=25, alpha=0.7, color=class_color[l])
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.savefig("./figs/top{}_numerical_value.pdf".format(label_size))
plt.show()
plt.close()


num_linear_model = NumericalModel(label_size)
num_linear_model, epoch_loss_list, epoch_accuracy_list = train(num_linear_model, train_text, train_numerical, train_labels,
                                                        test_text, test_numerical, test_labels, num_epochs=10)
num_latent = num_linear_model.numerical_module(all_numerical).numpy()
pca = PCA(n_components=2)
numerical_embedding = pca.fit_transform(num_latent)
plt.title("Numerical Latent Values", fontsize=15)
for l in range(label_size):
    tmp = numerical_embedding[np.where(all_labels==l)[0]]
    plt.scatter(tmp[:, 0], tmp[:, 1], label="Class {}".format(l), s=30, alpha=0.7, color=class_color[l])
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.savefig("./figs/top{}_numerical_latent_value.pdf".format(label_size))
plt.show()
plt.close()
# --------------------------------------------------------------------------------

pca = PCA(n_components=2)
text_embedding = pca.fit_transform(all_text)
plt.title("Text TF Vectors", fontsize=15)
for l in range(label_size):
    tmp = text_embedding[np.where(all_labels==l)[0]]
    plt.scatter(tmp[:, 0], tmp[:, 1], label="Class {}".format(l), s=30, alpha=0.7, color=class_color[l])
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.savefig("./figs/top{}_text_value.pdf".format(label_size))
plt.show()
plt.close()


text_linear_model = TextLinearModel(vocab_size, label_size)
text_linear_model, epoch_loss_list, epoch_accuracy_list = train(text_linear_model, train_text, train_numerical, train_labels,
                                                        test_text, test_numerical, test_labels, num_epochs=10)
text_latent = text_linear_model.text_module(all_text).numpy()
pca = PCA(n_components=2)
text_embedding = pca.fit_transform(text_latent)
plt.title("Text TF Latent Vectors", fontsize=15)
for l in range(label_size):
    tmp = text_embedding[np.where(all_labels==l)[0]]
    plt.scatter(tmp[:, 0], tmp[:, 1], label="Class {}".format(l), s=30, alpha=0.7, color=class_color[l])
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.savefig("./figs/top{}_text_linear.pdf".format(label_size))
plt.show()
plt.close()


text_cnn_model = TextCNNModel(vocab_size, label_size)
text_cnn_model, epoch_loss_list, epoch_accuracy_list = train(text_cnn_model, train_text, train_numerical, train_labels,
                                                        test_text, test_numerical, test_labels, num_epochs=10)
text_latent = text_cnn_model.text_module(all_text)
text_latent = text_cnn_model.text_dense(text_latent).numpy()
pca = PCA(n_components=2)
text_embedding = pca.fit_transform(text_latent)
plt.title("Text Word2Vec", fontsize=15)
for l in range(label_size):
    tmp = text_embedding[np.where(all_labels==l)[0]]
    plt.scatter(tmp[:, 0], tmp[:, 1], label="Class {}".format(l), s=30, alpha=0.7, color=class_color[l])
plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.savefig("./figs/top{}_text_cnn.pdf".format(label_size))
plt.show()
plt.close()
