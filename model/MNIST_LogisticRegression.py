import torch.nn as nn

class LogisticRegression_mnist(nn.Module):
    def __init__(self):
        super(LogisticRegression_mnist, self).__init__()
        self.linear = nn.Linear(784, 10)
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x