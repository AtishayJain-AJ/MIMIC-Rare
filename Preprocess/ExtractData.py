'''
Description:
    Extract data for model training.

Author:
    Jiaqi Zhang
'''

import pandas as pd
import numpy as np


def extractTopDataWithTrueValue():
    '''
    Extract data of the top 10 most common ICD-9 labels.
    '''
    note_data = pd.read_pickle("../data/notes_Demographics_labevents.pkl")
    note_data = note_data[["subject_id", "text", "icd9_code"]]
    note_data = note_data.groupby("subject_id").apply(lambda x: x.iloc[0]).reset_index(drop=True)

    lab_data = pd.read_pickle("../data/Columned_labevents.pkl")
    lab_data = lab_data.drop(columns=["index", "hadm_id"])
    lab_data = lab_data.groupby("subject_id").apply(lambda x: x.mean()).reset_index(drop=True)

    merged_data = pd.merge(note_data, lab_data, how="left").reset_index(drop=True)
    merged_data = merged_data[(~pd.isna(merged_data.text)) & (~pd.isna(merged_data.icd9_code))]
    print("Merged data shape : ", merged_data.shape)
    print("Merged data columns : ", merged_data.columns.values)

    # select data of the top 10 most common ICD9 code
    labels = merged_data.icd9_code
    label_sample_size = {each : len(np.where(labels == each)[0]) for each in np.unique(labels)}
    label_sample_size = [(each, label_sample_size[each]) for each in label_sample_size]
    label_sample_size = sorted(label_sample_size, key=lambda x: x[1])[::-1]
    idx = np.where(labels.apply(lambda x: x in [each[0] for each in label_sample_size[:10]]) == True)
    extracted_data = merged_data.iloc[idx].reset_index(drop=True)

    # fill in missing data
    extracted_data = extracted_data.drop(columns=["POTASSIUM"])
    numerical_feature_name = [
        'CREATININE', 'GLUCOSE', 'HEMOGLOBIN', 'LYMPHOCYTES',
        'MCH', 'PO2', 'PTT', 'RBC']
    nan_col_name = [each for each in numerical_feature_name if np.any(np.isnan(extracted_data[each].values))]
    for c in nan_col_name:
        extracted_data[c] = extracted_data[c].fillna(extracted_data[c].mean())
    print("Extracted data shape : ", extracted_data.shape)
    extracted_data.to_csv("../data/top10_label_data.csv")




if __name__ == '__main__':
    extractTopDataWithTrueValue()