import torch
class LogisticRegression_higgs(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression_higgs, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        torch.nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return torch.sigmoid(self.linear(x))