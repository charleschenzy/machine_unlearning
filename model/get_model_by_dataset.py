import torch.nn as nn
import torch.nn.functional as F
from model.CNN import *
from model.covtype_LogisticRegression import *
from model.HIGGS_LogisticRegression import *
from model.MNIST_LogisticRegression import *

def get_model(data_name):

    if data_name == 'cifar10' or data_name == 'cifar100':
        model = LeNet()
    elif data_name == 'mnist':
        model = LogisticRegression_mnist()
    elif data_name == 'covtype':
        model = LogisticRegression_cov()
    elif data_name == 'higgs':
        model = LogisticRegression_higgs(28)

    return model

