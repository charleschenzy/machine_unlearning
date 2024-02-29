import torch
from Load_data.processing import put_on_device,calculate_delta_data,create_unit_matrix_mnist
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


delta_H, v, I = create_unit_matrix_mnist('mnist')

def cal_H_L_BFGS(grad,params,B,delta_B,lr,n,m,args):
    #if flag ==
    #print(params[1])

    delta_W = put_on_device(calculate_delta_data(params),device)

    delta_G = put_on_device(calculate_delta_data(grad),device)

    print('len(delta_W) = {}'.format(len(delta_W)))

    for i in range(n,m):

        hessian_approx_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat = cal_approx_hessian_vec_prod0_3([delta_W[0],delta_W[1]], [delta_G[0],delta_G[1]], delta_H[i].to(device), 2, True, device)

        #delta_H.append(hessian_approx_prod)
        delta_H.append(torch.mm(I.to(device),delta_H[i].to(device)) - hessian_approx_prod * (lr/B-delta_B))

    return delta_H

def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, v_vec, k, is_GPU, device):
    zero_mat_dim = k

    curr_S_k = torch.t (torch.cat (list (S_k_list), dim=0))
    curr_Y_k = torch.t (torch.cat (list (Y_k_list), dim=0))


    curr_Y_k = curr_Y_k.to(device)
    curr_S_k = curr_S_k.to(device)

    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)


    if is_GPU:

        R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())

        L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)

    else:
        R_k = np.triu(S_k_time_Y_k.numpy())

        L_k = S_k_time_Y_k - torch.from_numpy(R_k)

    D_k_diag = torch.diag(S_k_time_Y_k)

    sigma_k = torch.mm(Y_k_list[-1], torch.t(S_k_list[-1])) / (torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))

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

