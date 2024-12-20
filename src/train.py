'''
dvc stage add -n train \
    -d data/prepare/train.csv \
    -d src/train.py \
    -p train.n_estim,train.depth \
    -o models/model.pkl \
    -M metrics/train_metrics.json \
    --plots plots/confusion_matrix.png \
    --plots plots/feature_importance.png \
    python src/train.py
'''

import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import json

'''
train.py: TRAIN MODEL IRIS DATASET USING NN WITH PYTORCH
'''


class Net(nn.Module):
    def __init__(self, fc2_fc3_n: int):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, fc2_fc3_n)
        self.fc3 = nn.Linear(fc2_fc3_n, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)
        return X

with open('params.yaml', 'r') as params_file:
    params = yaml.safe_load(params_file)
    
fc2_fc3_n = int(params['train']['fc2_fc3_n'])
epoch = int(params['train']['epoch'])
lr = float(params['train']['lr'])

# Load data
train_df = pd.read_csv('data/prepare/train.csv')

train_X = Variable(torch.Tensor(np.array(train_df.iloc[:, :-1])).float())
train_y = Variable(torch.Tensor(np.array(train_df.iloc[:, -1])).long())

net = Net(fc2_fc3_n)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Train the model + tqdm for progress bar
for epoch in tqdm.tqdm(range(epoch)):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    
predictions = net(train_X)
_, predicted = torch.max(predictions, 1)

# Save model
torch.save(net, 'models/model.pth')

# Calculate metrics and save them
accuracy = (train_y == predicted).sum().item() / len(train_y)
metrics = {'train_accuracy': accuracy}
with open('metrics/train_metrics.json', 'w') as metrics_file:
    json.dump(metrics, metrics_file)
    
# Save confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(train_y, predicted)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Save feature importance
importances = net.fc1.weight.data.abs().mean(dim=0).numpy()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feature_importance = dict(zip(features, importances))
plt.figure()
sns.barplot(x=list(feature_importance.keys()), y=list(feature_importance.values()))
plt.title('Feature importance')
plt.savefig('plots/feature_importance.png')
plt.close()

    

