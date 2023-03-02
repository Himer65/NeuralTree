import torch
from torch import nn

class NeuralTree(nn.Module):
    #Deep Neural Decision Tree
    def __init__(self,
                 features,
                 points,
                 classes,
                 temperature=0.1):
        super().__init__()
        self.features, self.points, self.classes, self.temperature = \
             features,      points,      classes,      temperature
        self.T = 1 / temperature
        self.beta = nn.Parameter(torch.rand(features, points))
        self.classifier = nn.Parameter(torch.rand((points+1)**features, classes))

    @property
    def b(self):
        beta = torch.cumsum(-self.beta.sort(1).values, 1)
        return torch.cat([torch.zeros(beta.size(0), 1, device=beta.device), beta], 1)

    def forward(self, x):
        if x.dim() > 2:
            out = self(x.flatten(0, -2))
            return out.view(*x.shape[:-1], out.size(-1))
        elif x.dim() == 1:
            return self(x.unsqueeze(0)).squeeze(0)

        x = x.unsqueeze(-1)
        w = torch.arange(1, self.points+2, dtype=torch.float, device=x.device).unsqueeze(0)
        hot = torch.softmax((x @ w + self.b) * self.T, -1).transpose(0, 1) #?
        leaves = hot[0]
        for i in hot[1:]:
            leaves = torch.einsum('bu,bi->bui', leaves, i)
            leaves = leaves.view(leaves.size(0), leaves.size(1)*leaves.size(2))
        out = leaves @ self.classifier
        return out

    def extra_repr(self):
        return f'features={self.features}, points={self.points}, classes={self.classes}, temperature={self.temperature}'

class NeuralRandomForest(nn.Module):
    def __init__(self,
                 features,
                 points,
                 classes,
                 numbers=10,
                 temperature=0.1):
        super().__init__()
        self.numbers, self.classes = numbers, classes
        self.trees  = nn.ModuleList()
        self.weight = nn.Parameter(torch.empty(numbers).uniform_(-0.01, 0.01))
        subset = []

        for _ in range(numbers):
            idx = self.index(features)
            m = NeuralTree(len(idx), points, classes, temperature)
            self.trees.append(m)
            subset.append(idx.tolist())        
        self.register_buffer('subset', torch.tensor(subset))

    def forward(self, x):
        w = F.softmax(self.weight, -1)
        out = 0
        for i, mod in enumerate(self.trees):
            inp = x[...,self.subset[i]]
            out += mod(inp) * w[i]
            
        return out

    def index(self, f):
        size = torch.randperm(min(f, 10))
        idx = torch.randperm(f)
        return idx[size]

    def __repr__(self):
        return f'{self.__class__.__name__}(numbers_of_trees={self.numbers}, classes={self.classes})'