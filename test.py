import pandas as pd
import torch
from torch import nn
from torch.nn.optim import Adam
from torch.nn import functional as F
from sklearn.preprocessing import LabelEncoder
from tree import NeuralRandomForest

def dataset(root):
    data = pd.read_csv(root)
    data.drop(columns='Id', inplace=True)
    data['Species'] = LabelEncoder().fit_transform(data['Species'])
    src = torch.from_numpy(data.iloc[:,:-1].to_numpy()).float()
    trg = torch.from_numpy(data.iloc[:,-1].to_numpy())
    trg = F.one_hot(trg, 3).float()
    
    return src, trg


if __name__ == '__main__':
    X, Y = dataset('Iris.csv')
    tree = nn.Sequential(
        NeuralRandomForest(4, 1, 3, 5),
        nn.Softmax(-1)
    )
    opt = Adam(tree.parameters(), lr=1e-2)

    for ep in range(1, 501):
        pred = tree(X)
        loss = F.cross_entropy(pred, Y)
        loss.backward(), opt.step(), opt.zero_grad()
     
        if ep % 100 == 0:
            print(f'({ep})loss: {loss.item():.4f}')
