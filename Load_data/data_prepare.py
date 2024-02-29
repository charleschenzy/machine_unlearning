import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from Load_data.parameters import hyperparam
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from My_Sampler.subsetSeqSampler import SubsetSeqSampler
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from locale import normalize
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os, sys
import bz2
import requests

def load_MNIST_data():
    torch.manual_seed(42)
    # Define transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST('/tmp/pycharm_project_642/git_ignore_folder', download=True, train=True, transform=transform)
    testset = datasets.MNIST('/tmp/pycharm_project_642/git_ignore_folder', download=True, train=False, transform=transform)

    # Extract 10% of the data with the same distribution as the original dataset

    return trainset,testset

def load_covtype_1():

    covtype_path = "/home/huangt/chenziy/covtype.data"

    data = pd.read_csv(covtype_path, header=None, delimiter=',')

    X = data.iloc[:, :-1].values

    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    args = hyperparam()
    # covtype = fetch_covtype()

    # X = covtype.data
    # y = covtype.target

    class covtypeDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X.values, dtype=torch.float)
            self.y = torch.tensor(y.values, dtype=torch.float)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # scaler = StandardScaler()

    # X_scaled = scaler.fit_transform(X)

    # X_train_tensor = torch.from_numpy(X_scaled).float()
    # y_train_tensor = torch.from_numpy(y-1).long()

    # X_test_tensor = torch.from_numpy(X_scaled).float()
    # y_test_tensor = torch.from_numpy(y-1).long()

    train_dataset = covtypeDataset(X_train, y_train)
    test_dataset = covtypeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Extract 10% of the data with the same distribution as the original dataset
    train_size = len(train_dataset)

    indices = list(range(train_size))

    np.random.shuffle(indices)

    split = int(np.floor(args.percentage * train_size))

    train_idx, valid_idx = indices[split:], indices[:split]

    np.random.shuffle(indices)

    return train_dataset,test_dataset,train_size,indices,args,train_idx,valid_idx

def load_covtype():

    covtype = fetch_covtype()

    X = covtype.data

    y = covtype.target

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train_tensor = torch.from_numpy(X_scaled).float()

    y_train_tensor = torch.from_numpy(y - 1).long()

    X_test_tensor = torch.from_numpy(X_scaled).float()

    y_test_tensor = torch.from_numpy(y - 1).long()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)

    testset = TensorDataset(X_test_tensor, y_test_tensor)

    return trainset, testset

def prepare_higgs_dataset():

    data = pd.read_csv("HIGGS.csv")

    X = data.iloc[:, 1:]

    y = data.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    class HIGGSDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    trainset = HIGGSDataset(X_train, y_train)

    testset = HIGGSDataset(X_test, y_test)

    return trainset,testset
def clean_sensor_data0(file_name, is_classification, num_features, split_id=None):

    Y_data = []

    X_data = []

    #     configs = load_config_data(config_file)
    #
    #     num_features = int(configs[file_name]['feature_num'])

    with open(file_name) as fp:
        line = fp.readline()
        cnt = 1

        while line:
            #             print("Line {}".format(cnt))

            contents = line.split(' ')

            if ':' not in contents[-1]:
                contents.pop()

            if '\n' in contents[-1]:
                contents[-1] = contents[-1][:-1]

            Y_data.append(float(contents[0]))

            data_map = {}

            for i in range(len(contents) - 1):
                id = contents[i + 1].split(':')[0]

                curr_content = float(contents[i + 1].split(':')[1])

                data_map[id] = curr_content

            curr_X_data = []

            for i in range(num_features):
                if str(i + 1) in data_map:
                    curr_X_data.append(data_map[str(i + 1)])
                else:
                    curr_X_data.append(0.0)

            #             print(cnt, curr_X_data)

            cnt = cnt + 1

            X_data.append(curr_X_data)

            line = fp.readline()
    #
    #             curr_X_data = []
    #
    #             if len(contents) < num_features + 1:
    #                 continue
    #
    #             for i in range(num_features):
    #
    #                 id = contents[i+1].split(':')[0]
    #
    #                 if int(id) != i+1:
    #                     break
    #
    #                 curr_X_data.append(float(contents[i+1].split(':')[1]))
    #
    #
    #             if len(curr_X_data) < num_features:
    #                 continue
    #
    #             X_data.append(curr_X_data)
    #
    #
    #             cnt += 1

    X_data = normalize(np.array(X_data))

    train_X_data = torch.tensor(X_data, dtype=torch.double)

    train_Y_data = torch.tensor(Y_data, dtype=torch.double)

    if torch.min(train_Y_data) != 0:
        train_Y_data = train_Y_data - 1

    #     print('unique_Y::', torch.unique(train_Y_data))

    train_Y_data = train_Y_data.view(-1, 1)

    #     print('Y_dim::', train_Y_data.shape)

    if split_id is None:
        return split_train_test_data(train_X_data, train_Y_data, 0.1, is_classification)
    else:
        return train_X_data[0:split_id], train_Y_data[0:split_id], train_X_data[split_id:], train_Y_data[split_id:]


def extended_by_constant_terms(X, extend_more_columns):

    X = torch.cat((X, torch.ones([X.shape[0], 1], dtype=torch.double)), 1)

    if extend_more_columns:
        X = torch.cat((X, torch.rand([X.shape[0], 500], dtype=torch.double) / 100), 1)

    return X

def prepare_higgs(git_ignore_folder):

    '''if not os.path.exists(git_ignore_folder):
        os.makedirs(git_ignore_folder)

    if not os.path.exists(git_ignore_folder + '/higgs'):
        os.makedirs(git_ignore_folder + '/higgs')

    curr_file_name = git_ignore_folder + '/higgs/HIGGS'

    if not os.path.exists(git_ignore_folder + '/higgs/HIGGS.bz2'):
        print('start downloading higgs dataset')
        url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2'
        r = requests.get(url, allow_redirects=True)

        open(curr_file_name + '.bz2', 'wb').write(r.content)
        print('end downloading higgs dataset')

        print('start uncompressing higgs dataset')
        zipfile = bz2.BZ2File(curr_file_name + '.bz2')  # open the file
        data = zipfile.read()  # get the decompressed data
        #             newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
        open(curr_file_name, 'wb').write(data)  # write a uncompressed file

        print('end uncompressing higgs dataset')'''

    num_feature = 28

    train_X, train_Y, test_X, test_Y = clean_sensor_data0(git_ignore_folder + 'higgs.zip/HIGGS.csv.gz', True, num_feature,
                                                              -500000)

        #         train_X, train_Y, test_X, test_Y = load_data_multi_classes(, , )

    train_Y = train_Y.view(-1)

    test_Y = test_Y.view(-1)

    train_X = extended_by_constant_terms(train_X, False)

    test_X = extended_by_constant_terms(test_X, False)

        #         torch.save(train_X, git_ignore_folder + 'noise_X')
        #
        #         torch.save(train_Y, git_ignore_folder + 'noise_Y')

    print(train_X.shape)

    print(test_X.shape)
        #         train_data = MNIST(git_ignore_folder + '/mnist',
        #                    download=True,
        #                    transform=transforms.Compose([
        # #                         transforms.Resize((32, 32)),
        #                        transforms.ToTensor()]))
        #
        #         test_data = MNIST(git_ignore_folder + '/mnist',
        #                       train=False,
        #                       download=True,
        #                       transform=transforms.Compose([
        # #                         transforms.Resize((32, 32)),
        #                           transforms.ToTensor()]))

    return train_X, train_Y.type(torch.LongTensor), test_X, test_Y.type(torch.LongTensor)

def prepare_cifar10_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return trainset,testset

def split_train_test_data(data_name):
    if data_name == 'mnist':

        trainset, testset = load_MNIST_data()

    elif data_name =='covtype':

        trainset,testset = load_covtype()

    elif data_name == 'higgs' :

        trainset,testset = prepare_higgs_dataset()

    elif data_name == 'cifar10':

        trainset,testset = prepare_cifar10_dataset()

    train_size = len(trainset)

    indices = list(range(train_size))

    np.random.shuffle(indices)

    args = hyperparam()

    split = int(np.floor(0.1 * train_size))

    train_idx, valid_idx = indices[split:], indices[:split]

    np.random.shuffle(indices)

    torch.save((train_idx, valid_idx), 'train_valid_idx_lgre.pt')

    np.random.shuffle(indices)

    train_sampler = SubsetSeqSampler(indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler,
                                              shuffle=False, drop_last=args.drop_last, num_workers=2, pin_memory=True)

    #testloader = torch.utils.data.DataLoader(trainset,batch_size=512, sampler=trainloader
    #iterloader = iter(trainloader)

    return train_idx, valid_idx, indices, trainloader


