import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.spatial.distance import cdist
from sklearn import svm
from Preprocess.siamese_preprocess import getTFData
from siamese_utils import TripletDataset, SingleDataset, TripletLoss, EmbeddingNet, TripletNet


train_text, test_text, train_label, test_label, vocab_dict = getTFData("data/Final_data.pkl")

#Training

batch_size = 2
train_dataset = TripletDataset(train_text, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_net = EmbeddingNet(train_text.shape[1]).to(device)
model = TripletNet(emb_net).to(device)
margin = 2
#model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss = TripletLoss(margin)
losses = []
model.train()
for epoch in range(150):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data_input1 = batch[0][0].float().to(device)
        data_input2 = batch[0][1].float().to(device)
        data_input3 = batch[0][2].float().to(device)
        out1, out2, out3 = model(data_input1, data_input2, data_input3)
        
        l2_lambda = 0.0001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()) #replace pow(2.0) with abs() for L1
 
    
        
        iteration_loss = loss(out1, out2, out3)
        total_loss = iteration_loss + l2_lambda * l2_norm
        total_loss.backward()
        optimizer.step()
        epoch_loss+= total_loss
    losses.append(epoch_loss)

#Testing

test_dataset = SingleDataset(test_text, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
model.eval()
for batch in test_loader:
    test_out = emb_net(batch.float().to(device))

train_dataset = SingleDataset(train_text, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=False)
model.eval()
for batch in train_loader:
    train_out=emb_net(batch.float().to(device))

num_class_accuracy = 5 #Adjust to change metric (if value is 5, the metric is top-5 accuracy)

#Using SVM 

cls = svm.LinearSVC(C=0.1)
cls.fit(train_out.cpu().detach().numpy(), train_label)

probs = cls.decision_function(test_out.cpu().detach().numpy())
best_n = np.argsort(-probs, axis=1)[:,:num_class_accuracy]

count = 0
for i in range(len(test_label)):
    if test_label[i] in cls.classes_[best_n][i]:
        count += 1
print ("Test accuracy with SVM = ",count/len(test_label))

#Using closest neighbour

euc_distances = cdist(test_out.cpu().detach().numpy(), train_out.cpu().detach().numpy())

closest_neighbours = np.argsort(euc_distances,axis=1)

closest_neighbour = closest_neighbours[:,0:20]

test_pred = train_label[closest_neighbour]

count = 0
for i in range(len(test_pred)):
    test_i = test_pred[i]
    _, idx = np.unique(test_i, return_index=True)
    test_i = test_i[np.sort(idx)]
    if test_label[i] in test_i[0:num_class_accuracy]:
        count += 1

print ("Test accuracy with closest neighbouts = ",count/len(test_label))