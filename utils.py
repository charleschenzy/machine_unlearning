import torch
import numpy as np
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def list_difference(lst1, lst2):
    return list(set(lst1) - set(lst2))
def test_intersection():
    assert intersection([1, 2, 3], [2, 3, 4]) == [2, 3]
    assert intersection([1, 2, 3], [4, 5, 6]) == []
    assert intersection([1, 2, 3], [1, 2, 3]) == [1, 2, 3]
    assert intersection([], []) == []
test_intersection()
def slice(list):
    sliced_list = []
    for i in range(0, len(list), 7850):  # 把delta_H每7850条数据分隔开，因为最初创建的是一维的list
        slice = list[i:i + 7850]
        sliced_list.append(slice)
    return sliced_list


def load_rcv1_train_data():
    with open('rcv1_train.binary', 'rb') as file:
        binary_data = file.read()
        data = np.frombuffer(binary_data, dtype=np.int32)
        trainset = torch.from_numpy(data)
    return trainset
def load_rcv1_test_data():
    with open('rcv1_test.binary','rb') as file:
        binary_data = file.read()
        data = np.frombuffer(binary_data,dtype=np.int32)
        testset = torch.from_numpy(data)
    return testset


def split_train_test_rcv1(X, Y, ratio):
    Y_labels = torch.unique(Y)
    print(1)
    all_selected_rows = []

    for Y_label in Y_labels:
        rids = (Y.view(-1) == Y_label).nonzero()

        curr_selected_num = int(rids.shape[0] * ratio)

        if curr_selected_num == 0:
            continue

        rid_rids = torch.tensor(list(np.random.choice(list(range(rids.shape[0])), curr_selected_num, replace=False)))

        all_selected_rows.append(rids[rid_rids])

    selected_rows = torch.cat(all_selected_rows, 0).view(-1)

    train_set_rids = torch.tensor(list(set(range(X.shape[0])) - set(selected_rows.numpy())))

    test_X = torch.zeros(selected_rows.shape[0], X.shape[1])

    test_Y = torch.zeros(selected_rows.shape[0], Y.shape[1])

    for i in range(selected_rows.shape[0]):
        rid = selected_rows[i]

        test_X[i] = X[rid]

        test_Y[i] = Y[rid]

    train_X = torch.zeros(train_set_rids.shape[0], X.shape[1])

    train_Y = torch.zeros(train_set_rids.shape[0], Y.shape[1])

    for i in range(train_set_rids.shape[0]):
        rid = train_set_rids[i]

        train_X[i] = X[rid]

        train_Y[i] = Y[rid]

    return train_X, train_Y, test_X, test_Y

import os

import torch


def get_model_para_shape_list(para_list):
    """
    use case:full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
    :param para_list:
    :return:
    """
    shape_list = []
    full_shape_list = []
    total_shape_size = 0
    for para in list (para_list):
        all_shape_size = 1
        for i in range (len (para.shape)):
            all_shape_size *= para.shape[i]
        total_shape_size += all_shape_size
        shape_list.append (all_shape_size)
        full_shape_list.append (para.shape)
    return full_shape_list, shape_list, total_shape_size


def get_all_vectorized_parameters1(para_list):
    """
    use case: expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
    :param para_list:
    :return:
    """
    res_list = []
    i = 0
    for param in para_list:
        res_list.append(param.data.view(-1))
        i += 1
    return torch.cat(res_list, 0).view(1,-1)

def get_devectorized_parameters(params, full_shape_list, shape_list):
    params = params.view(-1)
    para_list = []
    pos = 0
    for i in range(len(full_shape_list)):
        param = 0
        if len(full_shape_list[i]) >= 2:
            curr_shape_list = list(full_shape_list[i])
            param = params[pos: pos+shape_list[i]].view(curr_shape_list)
        else:
            param = params[pos: pos+shape_list[i]].view(full_shape_list[i])
        para_list.append(param)
        pos += shape_list[i]
    return para_list

def get_data_class_num_by_name(dataname):
    if dataname == 'MNIST' :
        return 10
    elif dataname == 'covtype' :
        return 7
    elif dataname == 'higgs' :
        return 2
    elif dataname == 'cifar10' :
        return 10
    elif dataname == 'cifar100':
        return 100

def compute_model_difference(model1, model2):
    """
    计算两个相同结构的 PyTorch 模型中每一层的参数差的 L2 范数，并求和输出。
    """
    with torch.no_grad():
        # 获取两个模型的参数字典
        state_dict1 = model1.state_dict ()
        state_dict2 = model2.state_dict ()

        # 初始化差的 L2 范数和
        l2_norm_list = []

        # 遍历两个参数字典，计算每一层的参数差的 L2 范数并求和
        for key in state_dict1.keys ():
            # 仅计算形状相同的参数差
            if state_dict1[key].shape == state_dict2[key].shape:
                param_diff = state_dict1[key] - state_dict2[key]
                l2_norm = torch.norm (param_diff, p=2)
                l2_norm_list.append(l2_norm.item ())

        # 输出差的 L2 范数和
        print ("两个模型参数差的 L2 范数和为:", l2_norm_list)

def convert_model_list_to_tensor_matrix(model_list):
    with torch.no_grad():
        params_list = []
        for model in model_list:
            params = []
            for param in model.parameters ():
                params.append (param.detach ().cpu ())
            params = get_all_vectorized_parameters1(params)
            params_list.append (params)
        params_list = torch.cat(params_list, dim=0)
    return params_list

def insert_avg_tensors(lst, insert_length):
    new_list = []
    for i in range(insert_length):
        if i == 0:
            left_tensor = lst[i]
            right_tensor = lst[i+1]
        elif i == len(lst) - 1:
            new_list.extend([lst[i]])
            break
        else:
            left_tensor = lst[i]
            right_tensor = lst[i+1]
        avg_tensor = (left_tensor + right_tensor) / 2.0
        new_list.extend([lst[i], avg_tensor])
    for i in range(insert_length, len(lst)):
        new_list.extend ([lst[i]])
    return new_list

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def reinitialize_model(model):
    for module in model.modules():
        # if isinstance(module, torch.nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0, std=0.01)
            # torch.nn.init.kaiming_normal_ (module.weight, mode='fan_out', nonlinearity='relu')
            # if module.bias is not None:
                # torch.nn.init.zeros_(module.bias)
        if isinstance(module, torch.nn.Conv2d):
            # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.normal_ (module.weight, mean=0, std=0.01)
            # if module.bias is not None:
            #     torch.nn.init.zeros_(module.bias)
        # Add additional conditions here for other layer types as needed...

    return model

def hyper_para_function(data_name,model,args):

    if data_name == 'mnist' or data_name == 'covtype':

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

        return criterion,optimizer,scheduler

    elif data_name == 'higgs':

        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        return criterion,optimizer,scheduler

    elif data_name == 'cifar10':

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

        return criterion.optimizer,scheduler


if __name__ == '__main__':
    test_list = [1,2,3,4,5,6,7]
    test_list = [torch.tensor(x) for x in test_list]
    test_list = insert_avg_tensors(test_list,3)
    print(test_list)