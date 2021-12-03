# MIMIC

## Data Preprocessing

Scripts for extracting data from MIMIC (```ExtractData.py```) and prepare for data training (```PrepareData.py```) 
are located in the ```Preprocess/``` directory.

The preprocessing operations are in the ```ExtractData.py``` file. 
For the subject with multiple notes, I just take the first note; 
for nan value in lab event, I impute them with the average value.


In the ```PrepareData.py```, each text is represented by the term frequency vector.


## Model

I implemented a simple model with linear layers in the ```Model/ExtractData.py```. 
The model contains two encoders, one for text data and one for numerical data. Two encoders are dense layers.
After encoding, latent embeddings of two type of data are concatenated and the 
model uses a dense layer for the classification.  

I'll implement a CNN model with embedding layer, hoping to see a improved performance.

## Training and Testing

Run the ```Running.py``` will train the model for 20 epochs on training data and evaluate it on the testing data.