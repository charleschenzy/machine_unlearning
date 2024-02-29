import argparse
from Machine_Unlearn import *
import sys, os
from utils import *
from model.get_model_by_dataset import *
#from HIGGS.exp import *


sys.path.append(os.path.abspath(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract a percentage of the data with the same distribution as the original dataset')

    parser.add_argument('--percentage', type=float, default = 0.1,  help='the percentage of data to extract')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--batch_size', type=int,default = 16384, help='batch size')

    parser.add_argument('--drop_last', type=bool, default=False, help='drop last')

    parser.add_argument('--device', type=int, default=0, help='device of cuda')

    parser.add_argument('--lr', type=float, default = 0.1,help='lr of SGD')

    parser.add_argument('--T', type=int, default = 50,help='number of epochs')

    parser.add_argument('--L2', type=float, default =0.001, help='L2_norm')

    parser.add_argument('--k', type=int,default = 10, help='the epochs of unlearned information need to be stored')

    parser.add_argument('--m',type=int,default = 2, help='the gradients of unlearned information need to be stored')

    parser.add_argument('--dataset', default = 'higgs',help="dataset to be used")

    args = parser.parse_args()

    data_name = args.dataset

    model = get_model(data_name)

    delta_H, v_vec, I = create_unit_matrix_mnist(args.dataset)

    n = 0

    m = len(delta_H)

    l = len(delta_H)

    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=True)

    criterion,optimizer,scheduler = hyper_para_function(data_name,model,args)

    w_T_star,delta_wT = machine_unlearn(model, delta_H, args, n, m, l,criterion, optimizer, scheduler)

    torch.save(delta_wT, 'delta_wT_'+data_name+'.pt')

    torch.save(w_T_star, 'w_T_star_'+data_name+'.pt')