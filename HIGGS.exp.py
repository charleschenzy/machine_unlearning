import pandas as pd
import copy
import numpy as np
import torch
import datetime
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from utils import intersection, list_difference,slice
import argparse
from model.HIGGS_LogisticRegression import LogisticRegression_higgs

parser = argparse.ArgumentParser(description='Extract a percentage of the data with the same distribution as the original dataset')
parser.add_argument('--percentage', type=float, default=0.15, help='the percentage of data to extract')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch_size', type=int, default=16384, help='batch size')
parser.add_argument('--drop_last', action='store_true', help='drop last')
parser.add_argument('--device', type=int, default=1, help='device of cuda')
parser.add_argument('--lr', type=float, default=0.1, help='lr of SGD')
parser.add_argument('--T', type=int, default=40, help='number of epochs')
parser.add_argument('--L2', type=float, default=0.005,help='L2_norm')
parser.add_argument('--k',type=int, default=10,help='the hyperparameter k')
args = parser.parse_args()

data = pd.read_csv("HIGGS.csv")
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('len(x_train) = {}'.format(len(X_train)))
print('len(x_test) = {}'.format(len(X_test)))
def calculate_grad_unlearn_data(total_grad, unlearn_data_id_list, model, device, optimizer, B, delta_B, now_lr, args):
    # with torch.no_grad():
    unlearn_sampler = SubsetSeqSampler(unlearn_data_id_list)
    unlearnloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=unlearn_sampler,
                                                shuffle=False, drop_last=False)
    model.train()
    unlearn_criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
    for data, target in unlearnloader:
        data = data.to(device)
        target = target.to(device)
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
def calculate_retain_data(batch_model, retain_data_id_last_batch,optimizer):  #计算全集和遗忘集的差集信息
    retain_sampler = SubsetSeqSampler(retain_data_id_last_batch)
    retain_dataloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,sampler=retain_sampler,
                                                   shuffle=False,drop_last=False)
    print('retain_dataloader={}'.format(len(retain_dataloader)))
    model = copy.deepcopy(batch_model)
    model.train()
    for data,target in retain_dataloader:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        retain_gradients = [param.grad.clone().detach() for param in model.parameters()]
        retain_params = [param.data.clone().detach().cpu() for param in model.parameters()]
    #    optimizer.step()
    model.zero_grad()
    return retain_gradients,retain_params
def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, v_vec, k, is_GPU, device):
#    print('cal_approx_hessian_vec_prod0_3正在运行')
    zero_mat_dim = k  # ids.shape[0]

    curr_S_k = torch.t (torch.cat (list (S_k_list), dim=0))
    curr_Y_k = torch.t (torch.cat (list (Y_k_list), dim=0))

#    print('curr_S_k = {}'.format(curr_S_k.shape))
#    print('curr_Y_k = {}'.format(curr_Y_k.shape))

    curr_Y_k = curr_Y_k.to(device)
    curr_S_k = curr_S_k.to(device)

    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)

#    print('S_k_time_Y_k = {}'.format(S_k_time_Y_k.shape))
#    print('S_k_time_S_k = {}'.format(S_k_time_S_k.shape))

    if is_GPU:

        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())

        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)

    else:
        R_k = np.triu(S_k_time_Y_k.numpy())

        L_k = S_k_time_Y_k - torch.from_numpy(R_k)

    D_k_diag = torch.diag(S_k_time_Y_k)
#    print('D_k_diag = {}'.format(D_k_diag.shape))
    sigma_k = torch.mm(Y_k_list[-1], torch.t(S_k_list[-1])) / (torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
#    print('sigmaK = {}'.format(sigma_k))
    #if sigma_k < mini_sigma:
    #    sigma_k = mini_sigma
    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim * 2, 1], dtype=torch.double, device=device)
    else:
        p_mat = torch.zeros([zero_mat_dim * 2, 1], dtype=torch.double)
#    print('v_vec = {}'.format(v_vec.shape))
    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
#    print('tmp = {}'.format(tmp.shape))
    p_mat[0:zero_mat_dim] = tmp
    sigma_k = sigma_k.to(device)
    p_mat[zero_mat_dim:zero_mat_dim * 2] = torch.mm(torch.t(curr_S_k), v_vec) * sigma_k
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim=1)
    lower_mat = torch.cat([L_k, sigma_k * S_k_time_S_k], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat = np.linalg.inv(mat.cpu().numpy())
    inv_mat = torch.from_numpy(mat)
    if is_GPU:
        inv_mat = inv_mat.to(device)
    p_mat = torch.mm(inv_mat, p_mat.float())
    approx_prod = sigma_k * v_vec - torch.mm(torch.cat([curr_Y_k, sigma_k * curr_S_k], dim=1), p_mat)
    return approx_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat

def update_list(delta_H, vector_u):   #对应论文 Update: ∆H[0].append([u1, ..., up])
    delta_H.append(vector_u)
    return delta_H
def concatenated_vector(vector1,vector2):    #用来把[[10,784],[10]] 展成7850×1的向量
    return torch.cat([torch.cat([vector1, vector2.unsqueeze(1)], dim=1) for i in range(1)], dim=0)
def put_on_device(list,device):   #把张量放到Gpu上
    for i in range(len(list)):
            list[i] = list[i].to(device)
    return list
def calculate_delta_data(list):   #计算list中相邻元素的差值
    differences = []
    differences.append(list[0])
    for i in range(1, len(list)):
        difference = list[i] - list[i - 1]

        differences.append(difference)

    return differences
def create_unit_matrix():
    '''生成一个7850×7850的单位矩阵，然后再把这个矩阵一条一条的分隔开，即7850个7850×1的向量'''
    unit_matrix = torch.eye(29)
    tensor_list = [unit_matrix[i, :].view(1, -1) for i in range(29)]
    delta_H = update_list_v_vec(tensor_list)
    v_vec = update_list_v_vec(tensor_list)

    return delta_H, v_vec,unit_matrix
def update_list_v_vec(v_vec):
    list = []
    for i in range(len(v_vec)):
        list.append(torch.t(v_vec[i]))
    return list

def cal_H_by_L_BFGS(grad,params,B,delta_B,lr,n,m):  #L-BFGS算法
    delta_W = put_on_device(calculate_delta_data(params),device)
    delta_G = put_on_device(calculate_delta_data(grad),device)
    print('len(delta_W) = {}'.format(len(delta_W)))
    for i in range(n,m):
        hessian_approx_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat = cal_approx_hessian_vec_prod0_3([delta_W[0],delta_W[1]], [delta_G[0],delta_G[1]], delta_H[i].to(device), 2, True, device)
        #delta_H.append(hessian_approx_prod)
        delta_H.append(torch.mm(I.to(device),delta_H[i].to(device)) - hessian_approx_prod * (lr/B-delta_B))

    return delta_H

delta_H,v_vec,I = create_unit_matrix()
class HIGGSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

n = 0
m = len(delta_H)
print('m = {}'.format(m))

delta_G = []
model_param_list = []
train_data_ids = []
nable_grad_u_list = []
hvp_nable_grad_list = []
unlearn_data_id_each_batch = []
retain_data_id_each_batch = []
'''s1表示展平后的张量'''
retain_grad_list_s1 = []
retain_param_list_s1 = []
model_grad_list = []
G_s1 = []

trainset = HIGGSDataset(X_train, y_train)
testset = HIGGSDataset(X_test, y_test)
train_size = len(trainset)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor(args.percentage * train_size))
train_idx, valid_idx = indices[split:], indices[:split]
np.random.shuffle(indices)
torch.save((train_idx, valid_idx), 'train_valid_idx_HIGGS.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

delta_H = put_on_device(delta_H,device)

input_dim = X_train.shape[1]
print('input_dim = {}'.format(input_dim))
model = LogisticRegression_higgs(input_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

np.random.shuffle(indices)
train_sampler = SubsetSeqSampler(indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler,
                                        shuffle=False, drop_last=args.drop_last, num_workers=2)
print(len(trainloader))
iterloader = iter(trainloader)

for i in range(args.T):   #训练阶段

    try:      #每次从trainloader中提取一个batch
        batch = next(iterloader)
    except StopIteration:
        iterloader = iter(trainloader)
        batch = next(iterloader)

    train_loss = 0.0
    model.train()
    correct = 0
    total = 0
    batch_idx = 0

    data,target = batch
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target) + args.L2 * torch.sum(model.linear.weight**2)
    loss.backward()
    gradients = [param.grad.clone().detach() for param in model.parameters()]
    model_param_list.append([param.data.clone().detach().cpu() for param in model.parameters()])
    #print('len(model_param_list) = {}'.format(len(model_param_list)))
    #print('len(model_param_list[0]) = {}'.format(len(model_param_list[0])))
    #print('model_param_list[0].shape = {}'.format(model_param_list[0][0].shape))
    optimizer.step()
    train_loss += loss.item() * data.size(0)
    predicted = (output > 0.5).float()
    correct += (predicted == target).sum().item()
    total += target.size(0)

    train_data_ids.append(
        train_sampler.indices[batch_idx * trainloader.batch_size:(batch_idx + 1) * trainloader.batch_size])
    # print(len(train_data_ids))
    unlearn_data_id_each_batch.append(intersection(train_data_ids[-1], valid_idx))
    # print(len(unlearn_data_id_each_batch))
    retain_data_id_each_batch.append(list_difference(train_data_ids[-1], unlearn_data_id_each_batch[-1]))
    # print(len(retain_data_id_each_batch))

    # 对应公式里面的字母
    delta_B = len(retain_data_id_each_batch[-1])
    B = len(train_data_ids[-1])
    now_lr = scheduler.get_lr()[-1]
    if i >= args.T-args.k-2:
        #求G
        G = calculate_grad_unlearn_data(gradients, unlearn_data_id_each_batch[-1], model,
                                             device, optimizer, B, delta_B, now_lr, args)
        #把G展成向量
        #print('len(G) = {}'.format(len(G)))
        #print('G = {}'.format(G[0].shape))
        G_s1.append(concatenated_vector(G[0],G[1]).flatten().unsqueeze(1))
        nable_grad_u_list.append(G)
        '''存储展平后的参数和梯度'''
        #if len(nable_grad_u_list)>1:
        retain_grads, retain_params = calculate_retain_data(batch_model, retain_data_id_each_batch[-1], optimizer)  #用上一轮的参数计算
        retain_grad_list_s1.append(concatenated_vector(retain_grads[0],retain_grads[1]).flatten().unsqueeze(0))
        retain_param_list_s1.append(concatenated_vector(retain_params[0],retain_params[1]).flatten().unsqueeze(0))
        if len(retain_grad_list_s1)>=2:
            delta_H = cal_H_by_L_BFGS(retain_grad_list_s1[-2:], retain_grad_list_s1[-2:], B,delta_B,now_lr,n,m)
            n += 28
            m += 28
    batch_model = copy.deepcopy(model)
    batch_idx += 1
    train_accuracy = correct / total
    print(f"Epoch [{i + 1}]\tLoss: {train_loss / len(trainloader.sampler)}\tTrain Accuracy: {train_accuracy * 100:.2f}%")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    torch.save(model.state_dict(), 'model_HIGGS.pt')
    torch.save(model_param_list, 'model_param_list_HIGGS.pt')
    torch.save(scheduler, 'scheduler_HIGGS.pt')
#    torch.save(nable_grad_u_list, 'nable_grad_u_list_HIGGS.pt')
#    torch.save(hvp_nable_grad_list, 'hvp_nable_grad_list_HIGGS.pt')
    torch.save(train_data_ids, 'train_data_ids_HIGGS.pt')
    torch.save(delta_H, 'delta_H_HIGGS.pt')

model.eval()
y_pred = model(torch.tensor(X_test.values, dtype=torch.float32).to(device))
y_pred = (y_pred > 0.5).cpu().numpy().flatten()
y_test = y_test.values.astype(int)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

print('存了多少轮的参数：{}'.format(len(retain_param_list_s1)))
#print('delta_W = {}'.format(len(delta_W)))
#print('delta_W[4].shape = {}'.format(delta_W[4].shape))

#对应伪代码里面的delta_wT = 0
delta_wT = torch.zeros_like(delta_H[0])
def slice(list):
    sliced_list = []
    for i in range(0, len(list), 29):  # 把delta_H每7850条数据分隔开，因为最初创建的是一维的list
        slice = list[i:i + 29]
        sliced_list.append(slice)
    return sliced_list

delta_H = slice(delta_H)
print('更新后的len(delta_H) = {}'.format(len(delta_H)))
#print('len(delta_H[0]) = {}'.format(len(delta_H[0])))

#消掉U = {u_1,u_2,u_3,u_4...u_n}
for r in range(1,args.k-1):
    sum = torch.zeros(29,29).to(device)
    for q in range(29):
        sum += torch.mm(delta_H[r][q],torch.t(delta_H[0][q]))
    delta_wT += torch.mm(sum.to(device),G_s1[r+1])

#对应论文里的Update: ∆wT : ∆wT ← ∆wT + ∆G(k − 1)
delta_wT += G_s1[args.k-1]
print(delta_wT.shape)

#再把delta_wT的size恢复为和参数的size一样
print(delta_wT.shape)
delta_wT = delta_wT.view(1,29)
params = delta_wT[:,:-1]
bias = delta_wT[:,-1]
delta_wT = [params,bias]

#print(len(delta_wT))
#print(delta_wT[0].shape)
#print(delta_wT[1].shape)
#对应论文Return: w∗T = wT + ∆wT
w_T_star = [model_param_list[-1][0].to(device) + delta_wT[0],model_param_list[-1][1].to(device) + delta_wT[1]]
#print(len(w_T_star))
#print(w_T_star[0].shape)
#print(w_T_star[1].shape)
torch.save(delta_wT, 'delta_wT_HIGGS.pt')
torch.save(w_T_star,'w_T_star_HIGGS.pt')