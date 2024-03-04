import torch
from utils import *

def put_on_device(list,device):
    for i in range(len(list)):
            list[i] = list[i].to(device)
    return list

def load_vector_u(delta_H, vector_u):

    delta_H.append(vector_u)

    return delta_H

def create_unit_matrix_mnist(data_name):

    if data_name == 'mnist' :
        unit_matrix = torch.eye(7850)

        tensor_list = [unit_matrix[i, :].view(1, -1) for i in range(7850)]

        delta_H = update_list_v_vec(tensor_list)

        v_vec = update_list_v_vec(tensor_list)

    elif data_name == 'covtype' :
        unit_matrix = torch.eye(385)
        tensor_list = [unit_matrix[i, :].view(1, -1) for i in range(385)]
        delta_H = update_list_v_vec(tensor_list)
        v_vec = update_list_v_vec(tensor_list)

    elif data_name == 'higgs':
        unit_matrix = torch.eye(29)
        tensor_list = [unit_matrix[i, :].view(1, -1) for i in range(29)]
        delta_H = update_list_v_vec(tensor_list)
        v_vec = update_list_v_vec(tensor_list)

    #elif data_name == 'cifar10' :


    return delta_H, v_vec,unit_matrix

def update_list_v_vec(v_vec):
    list = []
    for i in range(len(v_vec)):
        list.append(torch.t(v_vec[i]))

    return list

def calculate_delta_data(list):

    differences = []
    differences.append(list[0])
    for i in range(1,len(list)):

        difference = list[i] - list[i - 1]

        differences.append(difference)

    return differences

def data_list():
    nable_grad_u_list = []
    hvp_nable_grad_list = []
    unlearn_data_id_each_batch = []

    #retain_data = D-unlearn_data
    retain_data_id_each_batch = []


    model_param_list = []
    model_grad_list = []


    return nable_grad_u_list,hvp_nable_grad_list,unlearn_data_id_each_batch,retain_data_id_each_batch,model_param_list,model_grad_list

def s1_list():

    retain_grad_list_s1 = []

    retain_param_list_s1 = []

    G_s1 = []

    delta_G = []

    return retain_param_list_s1,retain_grad_list_s1,G_s1,delta_G

def concatenated_vector(vector1,vector2):

    return torch.cat([torch.cat([vector1, vector2.unsqueeze(1)], dim=1) for i in range(1)], dim=0)
