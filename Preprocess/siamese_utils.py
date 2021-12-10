import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

class SingleDataset(Dataset):
    def __init__(self, df, labels):
        self.data = df
        self.labels = labels
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample = self.data[idx]
            
        return sample

class ClassificationDataset(Dataset):
    def __init__(self, df, labels):
        self.data = df
        self.labels = labels
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample, label = self.data[idx], self.labels[idx]
            
        return sample ,label  
    
class TripletDataset(Dataset):
    def __init__(self, df, labels):
        self.data = df
        self.labels = labels
        
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]for label in self.labels_set}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample1, label1 = self.data[idx], self.labels[idx]
        positive_index = idx
        while positive_index == idx:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        sample2 = self.data[positive_index]
        sample3 = self.data[negative_index]
            
        return (sample1,sample2,sample3), []


class EmbeddingNet(nn.Module):
    def __init__(self, input_size):
        super(EmbeddingNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(input_size, 128),
                                nn.PReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(128, 64),
                                )

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)



class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(64, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))



class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()