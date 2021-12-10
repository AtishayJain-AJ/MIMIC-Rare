import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.spatial.distance import cdist
from sklearn import svm
from Preprocess.siamese_preprocess import getTFData
from siamese_utils import ClassificationDataset, SingleDataset, EmbeddingNet, ClassificationNet


train_text, test_text, train_label, test_label, vocab_dict = getTFData("data/Final_data.pkl")

#Training

batch_size = 2


unique_labels = np.unique(train_label)
unique_labels_dict = {unique_labels[i] : i for i in range(len(unique_labels))}

one_hot_train_label = np.zeros((len(train_label), len(unique_labels)))
for i in range(len(train_label)):
        one_hot_train_label[i][unique_labels_dict[train_label[i]]] = 1

one_hot_test_label = np.zeros((len(test_label), len(unique_labels)))
for i in range(len(test_label)):
        one_hot_test_label[i][unique_labels_dict[test_label[i]]] = 1

train_dataset = ClassificationDataset(train_text, one_hot_train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


num_classes = len(np.unique(test_label))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_net = EmbeddingNet(train_text.shape[1]).to(device)
model = ClassificationNet(emb_net, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss = torch.nn.CrossEntropyLoss()
losses = []
model.train()
for epoch in range(150):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data_input, target = batch
        data_input = data_input.float().to(device)
        target = target.float().to(device)
        out = model(data_input)
 
    
        
        iteration_loss = loss(out, target)
        total_loss = iteration_loss #+ l2_lambda * l2_norm
        total_loss.backward()
        optimizer.step()
        epoch_loss+= total_loss
    losses.append(epoch_loss)


#Testing
test_dataset = SingleDataset(test_text, one_hot_test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
model.eval()
for batch in test_loader:
    test_out = model(batch.float().to(device))

train_dataset = SingleDataset(train_text, one_hot_train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=False)
model.eval()
for batch in train_loader:
    train_out=model(batch.float().to(device))


num_class_accuracy = 5 #Adjust to change metric (if value is 5, the metric is top-5 accuracy)

best_n = np.argsort(-test_out.cpu().detach().numpy(), axis=1)[:,:num_class_accuracy]
test_label_num = np.argmax(one_hot_test_label, axis = 1)

count = 0
for i in range(len(test_label_num)):
    if test_label_num[i] in best_n[i]:
        count += 1


print ("Test accuracy = ",count/len(test_label))