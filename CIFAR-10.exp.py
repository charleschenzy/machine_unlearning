
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from math_calculate.cal_hession_approximate import cal_H_L_BFGS
from utils import intersection, list_difference, slice
from Load_data.data_prepare import *
from Load_data.processing import data_list, s1_list, calculate_delta_data, update_list_v_vec, \
    create_unit_matrix_mnist, put_on_device, concatenated_vector
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
nable_grad_u_list, hvp_nable_grad_list, unlearn_data_id_each_batch, retain_data_id_each_batch, model_param_list, model_grad_list = data_list()
retain_param_list_s1, retain_grad_list_s1, G_s1, delta_G = s1_list()
# Define a simple CNN with fixed convolutional layers
# 使用torch.nn包来构建神经网络.
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Freeze convolutional layers
        self.freeze_conv_layers()

    def forward(self, x):  # 正向传播过程
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

    def freeze_conv_layers(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False


# CIFAR-10 data loading and preprocessing

trainset,testset = prepare_cifar10_dataset()
train_loader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers = 2)
test_loader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=2)

# Initialize model, loss function, and optimizer
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# Training loop

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

def test_model(model, testloader, criterion, device):
    model.eval()

    test_loss = 0.0

    class_correct = list(0. for i in range(10))

    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data, target in testloader:

            data = data

            target = target

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)

            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            for i in range(len(target)):

                label = target.data[i]

                class_correct[label] += correct[i].item()

                class_total[label] += 1

    test_loss = test_loss / len(testloader.dataset)
    print('Test Accuracy: {:.2f}%'.format(100 * np.sum(class_correct) / np.sum(class_total)))
def calculate_grad_unlearn_data(total_grad, unlearn_data_id_list, model, device, optimizer, B, delta_B, now_lr,
                                args):
    # with torch.no_grad():
    unlearn_sampler = SubsetSeqSampler(unlearn_data_id_list)

    unlearnloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=unlearn_sampler,
                                                shuffle=False, drop_last=False)
    model.train()
    unlearn_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in unlearnloader:
        data = data

        target = target

        optimizer.zero_grad()

        output = model(data)

        loss = unlearn_criterion(output, target)

        loss.backward()

        unlearn_grad = [param.grad.clone().detach() for param in model.parameters()]

        with torch.no_grad():
            for i, layer_grad in enumerate(unlearn_grad):
                # print(now_lr)
                unlearn_grad[i] = (now_lr / (B - delta_B)) * layer_grad

                unlearn_grad[i] -= (now_lr * delta_B / (B - delta_B)) * total_grad[i]
    return unlearn_grad
def calculate_retain_data(batch_model, retain_data_id_last_batch, optimizer, criterion):

    retain_sampler = SubsetSeqSampler(retain_data_id_last_batch)

    retain_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=retain_sampler,
                                                    shuffle=False, drop_last=False)

    model = copy.deepcopy(batch_model)
    model.train()
    for data, target in retain_dataloader:
        data = data
        target = target

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        retain_gradients = [param.grad.clone().detach() for param in model.parameters()]

        retain_params = [param.data.clone().detach().cpu() for param in model.parameters()]
    #    optimizer.step()
    model.zero_grad()
    return retain_gradients, retain_params

args = hyperparam()
train_data_ids = []
delta_H, v_vec, I = create_unit_matrix_mnist('cifar10')

n = 0

m = len(delta_H)

l = len(delta_H)

#delta_H = put_on_device(delta_H, args.device)

train_idx, valid_idx, indices, trainloader = split_train_test_data('cifar10')

np.random.shuffle(indices)

train_sampler = SubsetSeqSampler(indices)

iterloader = iter(train_loader)

for epoch in range(args.T):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    try:
        batch = next(iterloader)

    except StopIteration:

        iterloader = iter(train_loader)
        batch = next(iterloader)

    data, target = batch
        # for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    print(data.shape)
    outputs = model(data)
    loss = criterion(outputs, target)+args.L2 * torch.sum(model.fc3.weight ** 2)
    loss.backward()

    gradients = [param.grad.clone().detach() if param.grad is not None else None for param in model.parameters()]
    print("gradients = ", gradients[0])
    model_param_list.append([param.data.clone().detach().cpu() for param in model.parameters()])
    print("len(model_param_list) = ", len(model_param_list))
    print("len(model_param_list[0]) = ", len(model_param_list[0]))
    print("model_param_list = ", model_param_list[0][0].shape)

    optimizer.step()

    train_loss += loss.item() * data.size(0)

    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == target).sum().item()

    total += target.size(0)

    train_data_ids.append(
        train_sampler.indices[batch_idx * trainloader.batch_size:(batch_idx + 1) * trainloader.batch_size])

    unlearn_data_id_each_batch.append(intersection(train_data_ids[-1], valid_idx))

    retain_data_id_each_batch.append(list_difference(train_data_ids[-1], unlearn_data_id_each_batch[-1]))

    delta_B = len(retain_data_id_each_batch[-1])

    B = len(train_data_ids[-1])

    now_lr = scheduler.get_last_lr()[-1]

    if epoch >= args.T - args.k - 2:
        print(1)
        G = calculate_grad_unlearn_data(gradients, unlearn_data_id_each_batch[-1], model,
                                            args.device, optimizer, B, delta_B, now_lr, args)
            # print('len(G) = {}'.format(len(G)))

            # print('G = {}'.format(G[0].shape))
        G_s1.append(concatenated_vector(G[0], G[1]).flatten().unsqueeze(1))

        nable_grad_u_list.append(G)

            # if len(nable_grad_u_list)>1:
        retain_grads, retain_params = calculate_retain_data(batch_model, retain_data_id_each_batch[-1], optimizer,
                                                                criterion)

        retain_grad_list_s1.append(concatenated_vector(retain_grads[0], retain_grads[1]).flatten().unsqueeze(0))

        retain_param_list_s1.append(concatenated_vector(retain_params[0], retain_params[1]).flatten().unsqueeze(0))

        if len(retain_grad_list_s1) >= args.m:
            delta_H = cal_H_L_BFGS(retain_grad_list_s1[-args.m:], retain_grad_list_s1[-args.m:], B, delta_B, now_lr,
                                       n, m)
            n += l

            m += l

    batch_model = copy.deepcopy(model)
    # Calculate average training loss per epoch
    train_loss /= len(train_loader.sampler)

    # Print training statistics after each epoch
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(
        epoch + 1,
        train_loss,
        100 * correct / total))
    test_model(model, test_loader, criterion, args.device)
    batch_idx += 1
        # Adjust learning rate
    scheduler.step()


print('Finished Training')

delta_wT = torch.zeros_like(delta_H[0])

delta_H = slice(delta_H)

for r in range(1, args.k - 1):

    sum = torch.zeros(l, l).to(args.device)

    for q in range(l):
        sum += torch.mm(delta_H[r][q], torch.t(delta_H[0][q]))

    delta_wT += torch.mm(sum, G_s1[r + 1])

delta_wT += G_s1[args.k - 1]

print(delta_wT.shape)

delta_wT = delta_wT.view(10, 785)

params = delta_wT[:, :-1]

bias = delta_wT[:, -1]

delta_wT = [params, bias]

w_T_star = [model_param_list[-1][0].to(args.device) + delta_wT[0],
            model_param_list[-1][1].to(args.device) + delta_wT[1]]

torch.save(delta_wT, 'delta_wT_CIFAR_10.pt')
torch.save(w_T_star, 'w_T_star_CIFAR_10.pt')
# Test the model
correct = 0
total = 0


