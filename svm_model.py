from sklearn import svm
from Preprocess.siamese_preprocess import getTFData

train_text, test_text, train_label, test_label, vocab_dict = getTFData("data/Final_data.pkl")

cls = svm.LinearSVC(C=0.1).fit(train_text, train_label)

num_class_accuracy = 5 #Adjust to change metric (if value is 5, the metric is top-5 accuracy)

probs = cls.decision_function(train_text)
best_n = np.argsort(-probs, axis=1)[:,:num_class_accuracy]

count = 0
for i in range(len(train_label)):
    if train_label[i] in rbf.classes_[best_n][i]:
        count += 1


print ("Test accuracy = ",count/len(test_label))