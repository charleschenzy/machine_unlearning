import torch
import torch.nn as nn
import torch.optim as optim

# 构建逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2输入特征，1输出特征

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 生成示例数据
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([[0], [1], [0]], dtype=torch.float32)

# 初始化模型和优化器
model = LogisticRegression()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for _ in range(1000):
    # 此处只进行一次迭代
    optimizer.zero_grad()
    output = model(X)
    loss = nn.BCELoss()(output, y)
    loss.backward()
    optimizer.step()

# 计算Hessian矩阵的大小
parameters = list(model.parameters())
hessian_matrix = torch.autograd.functional.hessian(loss, tuple(parameters))  # 考虑所有参数，包括偏置项
hessian_size = hessian_matrix.size()

print("Hessian矩阵的大小:", hessian_size)
