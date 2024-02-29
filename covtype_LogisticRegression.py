import torch.nn as nn

class LogisticRegression_cov(nn.Module):
    def __init__(self):
        super(LogisticRegression_cov, self).__init__()
        self.linear = nn.Linear(54, 7)
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x