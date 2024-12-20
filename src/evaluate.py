import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import json
from train import Net

'''
evaluate.py: TEST MODEL IRIS DATASET
'''

test_df = pd.read_csv('data/prepare/test.csv')
X_test = Variable(torch.Tensor(np.array(test_df.iloc[:, :-1])).float())
y_test = test_df.iloc[:, -1]

# Load trained
net = None
with open('models/model.pth', 'rb') as model_file:
    net = torch.load(model_file)
    
# accuracy
predictions = net(X_test)
_, predicted = torch.max(predictions, 1)

# metrics
accuracy = (predicted == y_test).sum().item() / len(y_test)
#save
with open('metrics/test_metrics.json', 'w') as accuracy_file:
    json.dump(accuracy, accuracy_file)
    
# roc as plot png
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test, predictions.detach().numpy()[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure()
color = ['aqua', 'darkorange', 'cornflowerblue']
for i in range(3):
    plt.plot(fpr[i], tpr[i], color=color[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('fp rate')
plt.ylabel('tp rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig('plots/roc.png')
plt.close()    
