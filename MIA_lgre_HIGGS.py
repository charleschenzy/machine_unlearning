# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
import time

import numpy as np
from torch.utils.data import Dataset
import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from membership_inference import train_attack_model, attack
from model.HIGGS_LogisticRegression import LogisticRegression_higgs
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
"""Step 0. Initialize Federated Unlearning parameters"""

data = pd.read_csv("HIGGS.csv", error_bad_lines=False)
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]   # 标签列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class HIGGSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Arguments ():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 20
        self.N_client = 20
        self.trainset = 'HIGGS'  # purchase, cifar10, mnist, adult
        self.data_name = 'HIGGS'
        self.forget_client_idx = 1
        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 1  # If want to forget, change None to the client index

def add_params_to_model(model_param_list, model):
    updated_models = []
    updated_model = copy.deepcopy(model)
    updated_model = updated_model.cuda()
    for i, params in enumerate(model_param_list):
        param_dict = updated_model.state_dict()
        for j, (name, param) in enumerate(updated_model.named_parameters()):
            param_dict[name] += params[j].cuda()
        updated_model.load_state_dict(param_dict)
        updated_models.append(param_dict)
    return updated_models

def test_model(model, testloader):
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    model.eval()
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    criteria = torch.nn.CrossEntropyLoss ()
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criteria(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(len(target)):
                label = target.data[i]
                label = int(label)
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    test_loss = test_loss/len(testloader.dataset)
    print('Test Accuracy: {:.2f}%'.format(100 * np.sum(class_correct) / np.sum(class_total)))
    return 100 * np.sum(class_correct) / np.sum(class_total)

def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Arguments ()
    torch.manual_seed (42)
    # kwargs for data loader
    trainset = HIGGSDataset(X_train, y_train)
    testset = HIGGSDataset(X_test, y_test)
    print (60 * '=')
    print ("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    #init_global_model = FMNIST_CNN()
    input_dim = X_train.shape[1]
    init_global_model = LogisticRegression_higgs (input_dim).to(device)
    model_para = torch.load('model_HIGGS'+'.pt')
    init_global_model.load_state_dict(model_para)
    test_loader = DataLoader(testset, batch_size=2048, shuffle=True)
    train_idx, valid_idx = torch.load('./train_valid_idx_HIGGS' + '.pt')
    train_sampler = SubsetRandomSampler (train_idx)
    print(len(train_sampler))
    unlearn_sampler = SubsetRandomSampler (valid_idx)
    print(len(unlearn_sampler))
    trainloader = torch.utils.data.DataLoader (trainset, batch_size=16384, sampler=train_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)

    unlearnloader = torch.utils.data.DataLoader (trainset, batch_size=16384, sampler=unlearn_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)
    print('len(trainloader) = {}'.format(len(trainloader)))
    print('len(unlearnloader) = {}'.format(len(unlearnloader)))
    client_loaders = [trainloader, unlearnloader]
    delta_wT = torch.load ('./delta_wT_HIGGS' + '.pt')
    delta_wT = [[delta_wT[0], delta_wT[1]]]
    unlearn_params_list = add_params_to_model(delta_wT, init_global_model)
#    init_global_model.load_state_dict (model_para)
    print(len(unlearn_params_list))
#    print(unlearn_params_list[1])

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print (60 * '=')
#    print ("Step4. Membership Inference Attack aganist GM...")
    # MIA setting:Target model == Shadow Model
    attack_model = train_attack_model (init_global_model, client_loaders, test_loader, args)
    test_model(init_global_model, test_loader)
    print ("Attacking against not unlearn  ")
    pre,rec = attack (init_global_model, attack_model, client_loaders, test_loader, args)
    update_global_model = LogisticRegression_higgs(input_dim).to(device)
    for i in range(len(unlearn_params_list)):
        update_global_model.load_state_dict(unlearn_params_list[i])
        print("\nEpoch  = {}".format(i))
        pre,rec = attack (update_global_model, attack_model, client_loaders, test_loader, args)
        test_model (update_global_model, test_loader)

if __name__ == '__main__':
    Federated_Unlearning ()