import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk

from Preprocess.PrepareData import _preprocessText, _padding


def getTFData(filename):
    '''
    Prepare data for mode training and testing.
    '''

    # Step 1: load all data
    data = pd.read_pickle(filename)
    #data = data.sample(frac=1).reset_index(drop=True)
    text_data = data.text.values
    labels = data.icd9_code.values
    #print("Data shape : ", data.shape)

    # Step 2: tokenize the data
    # text_tokens = [[each.strip() for each in sentence.split(" ")] for sentence in text_data]
    text_tokens = [_preprocessText(sentence) for sentence in text_data]

    # Step 3: construct the vocab dictionary.
    vocab_dict = {each: idx for idx, each in enumerate(np.unique(np.concatenate(text_tokens)))}
    text_idx = [[vocab_dict[each] for each in sentence] for sentence in text_tokens]
    text_encode = np.zeros((len(text_idx), len(vocab_dict.keys())))
    for i in range(len(text_idx)):
        text_encode[i][text_idx[i]] += 1
        
        
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    usable_labels = []

    for i in label_counts:
        if label_counts[i] >= 5:
            usable_labels.append(i)


    # Step 4: split train and test data
    train_idx = []
    test_idx = []
    for label in usable_labels:
        indices = np.squeeze(np.argwhere(labels==label))
        
        test_idx.extend(indices[0:2])
        train_idx.extend(indices[2:5])

    train_text = text_encode[train_idx]
    test_text = text_encode[test_idx]
    train_label = labels[train_idx]
    test_label = labels[test_idx]
    return train_text, test_text, train_label, test_label, vocab_dict



