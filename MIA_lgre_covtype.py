
import time

import numpy as np
# %%
import torch
import copy
# ourself libs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler

from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from membership_inference import train_attack_model, attack
from model.covtype_LogisticRegression import LogisticRegression_cov
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
"""Step 0. Initialize Federated Unlearning parameters"""

covtype = fetch_covtype()

X = covtype.data
y = covtype.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_tensor = torch.from_numpy(X_scaled).float()

y_train_tensor = torch.from_numpy(y-1).long()

X_test_tensor = torch.from_numpy(X_scaled).float()

y_test_tensor = torch.from_numpy(y-1).long()

class Arguments ():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 20
        self.N_client = 20
        self.trainset = 'covtype'  # mnist, cifar10, HIGGS, covtype
        self.data_name = 'covtype'
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
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
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
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    test_loss = test_loss/len(testloader.dataset)
    print('Test Accuracy: {:.2f}%'.format(100 * np.sum(class_correct) / np.sum(class_total)))
    return 100 * np.sum(class_correct) / np.sum(class_total)

def MIA_attack():
    """Step 1.Set the parameters for Federated Unlearning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Arguments ()

    torch.manual_seed (42)
    # kwargs for data loader
    print (60 * '=')
    print ("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    #init_global_model = FMNIST_CNN()
    init_global_model = LogisticRegression_cov ().to(device)
    # model_para, model_param_list, scheduler, nable_grad_u_list, hvp_nable_grad_list, train_data_ids = load_parameters(FL_params)
    # 加载 model_param_list
    model_para = torch.load('model_cov'+'.pt')

    init_global_model.load_state_dict(model_para)
#    transform = transforms.Compose ([transforms.ToTensor (),
#                                    transforms.Normalize ((0.1307,), (0.3081,))])

    trainset = TensorDataset(X_train_tensor, y_train_tensor)

    testset = TensorDataset(X_test_tensor, y_test_tensor)

    test_loader = DataLoader(testset, batch_size=2048, shuffle=True)

    train_idx, valid_idx = torch.load('./train_valid_idx_lgre_covtype' + '.pt')
    train_sampler = SubsetRandomSampler (train_idx)

    unlearn_sampler = SubsetRandomSampler (valid_idx)

    trainloader = torch.utils.data.DataLoader (trainset, batch_size=16384, sampler=train_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)

    unlearnloader = torch.utils.data.DataLoader (trainset, batch_size=16384, sampler=unlearn_sampler,
                                               shuffle=False, drop_last=False, num_workers=2)

    client_loaders = [trainloader, unlearnloader]

    delta_wT = torch.load ('./delta_wT_covtype' + '.pt')
    delta_wT = [[delta_wT[0], delta_wT[1]]]

    unlearn_params_list = add_params_to_model(delta_wT, init_global_model)
#    init_global_model.load_state_dict (model_para)

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print (60 * '=')

 #   for i in range(epoch):

    print ("Step4. Membership Inference Attack aganist GM...")
    # MIA setting:Target model == Shadow Model
    attack_model = train_attack_model (init_global_model, client_loaders, test_loader, args)

    test_model(init_global_model, test_loader)

    print ("Attacking against not unlearn  ")

    pre,rec = attack (init_global_model, attack_model, client_loaders, test_loader, args)

    update_global_model = LogisticRegression_cov().to(device)

    for i in range(len(unlearn_params_list)):
        update_global_model.load_state_dict(unlearn_params_list[i])

        #if i % 36 == 0:
        print("\nEpoch  = {}".format(i))

        pre,rec = attack (update_global_model, attack_model, client_loaders, test_loader, args)
        test_model (update_global_model, test_loader)


if __name__ == '__main__':
    MIA_attack()