import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def getTFData(filename):
    '''
    Prepare data for mode training and testing.
    '''
    numerical_feature_name = ['CREATININE', 'GLUCOSE', 'HEMOGLOBIN',
                              'LYMPHOCYTES','MCH', 'PO2', 'PTT', 'RBC']
    # Step 1: load all data
    data = pd.read_csv(filename)
    data = data[~pd.isna(data.text)]
    text_data = data.text.values
    labels = data.icd9_code.values
    numerical_Data = data[numerical_feature_name].values
    print("Data shape : ", data.shape)

    # Step 2: tokenize the data
    text_tokens = [[each.strip() for each in sentence.split(" ")] for sentence in text_data]

    # Step 3: construct the vocab dictionary.
    vocab_dict = {each: idx for idx, each in enumerate(np.unique(np.concatenate(text_tokens)))}
    text_idx = [[vocab_dict[each] for each in sentence] for sentence in text_tokens]
    text_encode = np.zeros((len(text_idx), len(vocab_dict.keys())))
    for i in range(len(text_idx)):
        text_encode[i][text_idx[i]] += 1

    # Step 4: convert label to term frequency coding
    unique_labels = np.unique(labels)
    unique_labels_dict = {unique_labels[i] : i for i in range(len(unique_labels))}
    one_hot_labels = np.zeros((len(labels), len(unique_labels)))
    for i in range(len(labels)):
        one_hot_labels[i][unique_labels_dict[labels[i]]] = 1

    # Step 4: split train and test data
    train_idx, test_idx = train_test_split(np.arange(data.shape[0], ), train_size=0.7)
    train_text = text_encode[train_idx]
    test_text = text_encode[test_idx]
    train_numerical = numerical_Data[train_idx]
    test_numerical = numerical_Data[test_idx]
    train_label = one_hot_labels[train_idx]
    test_label = one_hot_labels[test_idx]
    return train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict

# -----------------------------------------------

def _padding(text, max_langth):
    padding_seq = ["<PAD>" for _ in range(max_langth-len(text))]
    return text + padding_seq


def getIdxData(filename):
    '''
    Prepare data for mode training and testing.
    '''
    numerical_feature_name = ['CREATININE', 'GLUCOSE', 'HEMOGLOBIN',
                              'LYMPHOCYTES','MCH', 'PO2', 'PTT', 'RBC']
    # Step 1: load all data
    data = pd.read_csv(filename)
    data = data[~pd.isna(data.text)]
    text_data = data.text.values
    labels = data.icd9_code.values
    numerical_data = data[numerical_feature_name].values
    print("Data shape : ", data.shape)

    # Step 2: tokenize the data
    text_tokens = [[each.strip() for each in sentence.split(" ")] for sentence in text_data]
    max_length = np.max([len(each) for each in text_tokens])
    # text_tokens = np.asarray([_padding(each, max_length) for each in text_tokens if len(each) > 10])
    text_tokens = np.asarray([_padding(each, max_length) for each in text_tokens])

    # Step 3: construct the vocab dictionary.
    vocab_dict = {each: idx for idx, each in enumerate(np.unique(np.concatenate(text_tokens)))}
    text_idx = np.asarray([[vocab_dict[each] for each in sentence] for sentence in text_tokens])

    # Step 4: convert label to term frequency coding
    unique_labels = np.unique(labels)
    unique_labels_dict = {unique_labels[i] : i for i in range(len(unique_labels))}
    one_hot_labels = np.zeros((len(labels), len(unique_labels)))
    for i in range(len(labels)):
        one_hot_labels[i][unique_labels_dict[labels[i]]] = 1

    # Step 4: split train and test data
    train_idx, test_idx = train_test_split(np.arange(data.shape[0], ), train_size=0.7)
    train_text = text_idx[train_idx]
    test_text = text_idx[test_idx]
    train_numerical = numerical_data[train_idx]
    test_numerical = numerical_data[test_idx]
    train_label = one_hot_labels[train_idx]
    test_label = one_hot_labels[test_idx]
    return train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict


if __name__ == '__main__':
    # train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = \
    #     getTFData("../data/top10_label_data.csv")
    train_text, test_text, train_numerical, test_numerical, train_label, test_label, vocab_dict = \
        getIdxData("../data/top10_label_data.csv")
    print("Vocabulary size = {}".format(len(vocab_dict.keys())))
