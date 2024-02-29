import copy
import datetime

from torch.utils.data import DataLoader
from math_calculate.cal_hession_approximate import cal_H_L_BFGS

from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from utils import *
from Load_data.parameters import hyperparam
from Load_data.data_prepare import load_MNIST_data,split_train_test_data
from Load_data.processing import data_list, s1_list, calculate_delta_data, update_list_v_vec, \
    create_unit_matrix_mnist, put_on_device, concatenated_vector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

n_class_dict = dict ()
n_class_dict['mnist'] = 10
n_class_dict['cifar10'] = 10
n_class_dict['covtype'] = 7
n_class_dict['HIGGS'] = 1
n_class_dict['cofar100'] = 100
# Determine the number of classifications based on the dataset

args = hyperparam()

trainset, testset = load_MNIST_data()
num_class_featrues = 0

nable_grad_u_list, hvp_nable_grad_list, unlearn_data_id_each_batch, retain_data_id_each_batch, model_param_list, model_grad_list = data_list()

def test_model(model, testloader, criterion, device):
    model.eval()

    test_loss = 0.0

    class_correct = list(0. for i in range(10))

    class_total = list(0. for i in range(10))

    with torch.no_grad():

        for data, target in testloader:

            data = data.to(device)

            target = target.to(device)

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

retain_param_list_s1, retain_grad_list_s1, G_s1, delta_G = s1_list()
def calculate_grad_unlearn_data(total_grad, unlearn_data_id_list, model, device, optimizer, B, delta_B, now_lr,
                                args):
    # with torch.no_grad():
    unlearn_sampler = SubsetSeqSampler(unlearn_data_id_list)

    unlearnloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=unlearn_sampler,
                                                shuffle=False, drop_last=False)
    model.train()
    unlearn_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
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
def calculate_retain_data(batch_model, retain_data_id_last_batch, optimizer, criterion):

    retain_sampler = SubsetSeqSampler(retain_data_id_last_batch)

    retain_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=retain_sampler,
                                                    shuffle=False, drop_last=False)

    model = copy.deepcopy(batch_model)

    model.train()

    for data, target in retain_dataloader:
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        retain_gradients = [param.grad.clone().detach() for param in model.parameters()]

        retain_params = [param.data.clone().detach().cpu() for param in model.parameters()]
    #    optimizer.step()
    model.zero_grad()
    return retain_gradients, retain_params

def machine_unlearn(model, delta_H, args, n, m, l, criterion, optimizer, scheduler):

    train_data_ids = []

    data_name = args.dataset

    num_class = get_data_class_num_by_name(data_name)

    delta_H = put_on_device(delta_H, args.device)

    train_idx, valid_idx, indices, trainloader = split_train_test_data(data_name)

    np.random.shuffle(indices)

    train_sampler = SubsetSeqSampler(indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler,
                                              shuffle=False, drop_last=args.drop_last, num_workers=2, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size = 2048, shuffle=True)
    iterloader = iter(trainloader)

    print('batch size for each epoch = {}'.format(args.batch_size))

    for i in range(args.T):

        try:
            batch = next(iterloader)

        except StopIteration:

            iterloader = iter(trainloader)
            batch = next(iterloader)

        train_loss = 0.0
        model.train()
        correct = 0
        total = 0
        batch_idx = 0

        data, target = batch

        print('target = {}'.format(target.shape[0]))

        data = data.to(args.device)

        target = target.to(args.device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target) + args.L2 * torch.sum(model.linear.weight ** 2)
        loss.backward()

        gradients = [param.grad.clone().detach() for param in model.parameters()]
        
        model_param_list.append([param.data.clone().detach().cpu() for param in model.parameters()])

        optimizer.step()

        train_loss += loss.item() * data.size(0)

        _, predicted = torch.max(output.data, 1)

        total += target.size(0)

        correct += (predicted == target).sum().item()

        print('total = {}'.format(total))

        print('correct = {}'.format(correct))

        train_data_ids.append(
            train_sampler.indices[batch_idx * trainloader.batch_size:(batch_idx + 1) * trainloader.batch_size])

        unlearn_data_id_each_batch.append(intersection(train_data_ids[-1], valid_idx))

        retain_data_id_each_batch.append(list_difference(train_data_ids[-1], unlearn_data_id_each_batch[-1]))

        delta_B = len(retain_data_id_each_batch[-1])

        B = len(train_data_ids[-1])

        now_lr = scheduler.get_lr()[-1]

        if i >= args.T - args.k - 2:

            G = calculate_grad_unlearn_data(gradients, unlearn_data_id_each_batch[-1], model,
                                            args.device, optimizer, B, delta_B, now_lr, args)

            # print('len(G) = {}'.format(len(G)))

            #print('G = {}'.format(G[0].shape))

            G_s1.append(concatenated_vector(G[0], G[1]).flatten().unsqueeze(1))

            num_class_featrues = concatenated_vector(G[0], G[1]).size()[1]

            #print('num_class = {}'.format(num_class))
            #print('G_s1[0][0].shape = {}'.format(G_s1[0][0].shape))

            nable_grad_u_list.append(G)

            # if len(nable_grad_u_list)>1:
            retain_grads, retain_params = calculate_retain_data(batch_model, retain_data_id_each_batch[-1], optimizer,criterion)

            retain_grad_list_s1.append(concatenated_vector(retain_grads[0], retain_grads[1]).flatten().unsqueeze(0))

            retain_param_list_s1.append(concatenated_vector(retain_params[0], retain_params[1]).flatten().unsqueeze(0))

            if len(retain_grad_list_s1) >= args.m:

                delta_H = cal_H_L_BFGS(retain_param_list_s1[-args.m:], retain_grad_list_s1[-args.m:], B, delta_B, now_lr,
                                       n, m,args)
                n += l

                m += l

        batch_model = copy.deepcopy(model)

        # hvp_nable_grad_list = calculate_hessian(model, retain_data_id_each_batch[-1], hvp_nable_grad_list, now_lr)
        # hvp_nable_grad_list.append(copy.deepcopy(grad_u))

        batch_idx += 1

        train_loss = train_loss / len(trainloader.sampler)

        test_model(model, testloader, criterion, args.device)

        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(
            i + 1,
            train_loss,
            100 * correct / total))
        scheduler.step()

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        torch.save(model.state_dict(), 'model_'+data_name+'.pt')
        # torch.save(model_param_list, 'model_param_list_mnist.pt')
        torch.save(scheduler, 'scheduler_'+data_name+'.pt')
        # torch.save(nable_grad_u_list, 'nable_grad_u_list_mnist.pt')
        # torch.save(hvp_nable_grad_list, 'hvp_nable_grad_list_mnist.pt')
        torch.save(train_data_ids, 'train_data_ids_'+data_name+'.pt')
        torch.save(delta_H, 'delta_H_'+data_name+'.pt')

    print('the stored parameter rounds: {}'.format(len(retain_param_list_s1)))

    # print('delta_W = {}'.format(len(delta_W)))
    # print('delta_W[4].shape = {}'.format(delta_W[4].shape))

    delta_wT = torch.zeros_like(delta_H[0])

    delta_H = slice(delta_H)

    for r in range(1, args.k - 1):

        sum = torch.zeros(l, l).to(args.device)

        for q in range(l):

            sum += torch.mm(delta_H[r][q], torch.t(delta_H[0][q]))

        delta_wT += torch.mm(sum, G_s1[r + 1])

    delta_wT += G_s1[args.k - 1]

    print(delta_wT.shape)

    delta_wT = delta_wT.view(num_class, num_class_featrues)

    params = delta_wT[:, :-1]

    bias = delta_wT[:, -1]

    delta_wT = [params, bias]

    w_T_star = [model_param_list[-1][0].to(args.device) + delta_wT[0], model_param_list[-1][1].to(args.device) + delta_wT[1]]

    #torch.save(delta_wT, 'delta_wT_mnist.pt')
    #torch.save(w_T_star, 'w_T_star_mnist.pt')

    return w_T_star,delta_wT



    #testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=True)

    #train_data_ids = []
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print(device)
    #model = LogisticRegression().to(device)

    #criterion = torch.nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    #valid_loss_min = np.Inf












