import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat
import torch.optim as optim
import torch
import numpy as np
from utils.training import train
from utils.testing import test
from utils.aggregation import communication
from adaptation import LMMDLoss



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BrainTumor')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='./data/')
    parser.add_argument('--iters', type=int, default=50,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--net', type=str, default='ViT-B/16',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[3])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--step', type=float, default=0)
    parser.add_argument('--aggmode', type=str, default='att')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--temp', type=float, default=0.5)
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args.n_clients = 4
    args = img_param_init(args)
    os.makedirs('./data/', exist_ok=True)
    server_model = ClipModelat(
        args.net, attention=True, freezepy=True)
    train_loaders, val_loaders, test_loaders, test_train, train_test_loaders = get_data(
        args.dataset)(args, server_model)
    server_model.initdgatal(test_loaders[1])
    # For multi client
    l1, l2, l3 = len(train_test_loaders[0]), len(train_test_loaders[1]), len(train_test_loaders[2])
    l1 = l1 * args.batch
    l2 = l2 * args.batch
    l3 = l3 * args.batch
    l = []
    l.append(l1)
    l.append(l2)
    l.append(l3)
    client_num = len(test_loaders)
    sclient_num = client_num-len(args.test_envs)
    client_weights = [l[i]/(l1+l2+l3) for i in range(sclient_num)]
    # client_weights = [1.0]
    client_weights.append(0.33333333333)
    print(client_weights)
    models = [copy.deepcopy(server_model)for idx in range(client_num)]
    for i in range(client_num):
        models[i].model.to(device)
        models[i].fea_attn.to(device)
    best_changed = False
    server_model_pre = server_model
    # best_acc = [0. for j in range(client_num)]
    best_acc = 0
    finalrecord = ''
    logrecord = ''
    log = []
    log2 = []
    previous_nets = models
    # optimizers = [optim.Adam(params=[{'params': models[idx].parameters()}], lr=args.lr, betas=(
        # args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
    optimizers = [optim.Adam(params=[{'params': models[idx].fea_attn.parameters()}], lr=args.lr, betas=(
        args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
    for a_iter in range(args.iters): #All Epoch
        mmd_loss = LMMDLoss()
        for wi in range(args.wk_iters): #Each client local training epoch
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else: #Client i finish training procedure
                    train(
                        args, model, train_loaders[client_idx], optimizers[client_idx], device, test_train[0], mmd_loss, server_model_pre, previous_nets[client_idx])
                    args.step += 1


        with torch.no_grad():
            server_model_pre = server_model
            previous_nets = models
            server_model, models = communication(
                args, server_model, models, client_weights)

            val_acc_list = [0. for j in range(client_num)]
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    train_acc = test(
                        args, model, train_test_loaders[client_idx], device)
                    print(' Site-{:d}| Train Acc: {:.4f}'.format(
                        client_idx, train_acc))
                    logrecord += ' Site-{:d}| Train Acc: {:.4f}\n'.format(
                        client_idx, train_acc)

                    val_acc = test(
                        args, model, val_loaders[client_idx], device)
                    val_acc_list[client_idx] = val_acc
                    print(' Site-{:d}| Val  Acc: {:.4f}'.format(
                        client_idx, val_acc), flush=True)
                    logrecord += ' Site-{:d}| Val  Acc: {:.4f}\n'.format(
                        client_idx, val_acc)
            # best_acc = 0
            test_acc_list = [0. for j in range(client_num)]
            for client_idx in range(client_num):
                if client_idx in args.test_envs:
                    # print(client_idx) # test is site 0
                    test_acc, bacc = test(args, server_model,
                                    test_loaders[client_idx], device)
                    log.append([test_acc])
                    log2.append([bacc])
                    if test_acc > best_acc:
                        best_acc = test_acc
                        # torch.save(server_model, './model/Ours_Skin2.pt')
                else:
                    test_acc, bacc = test(
                        args, models[client_idx], test_loaders[client_idx], device)
                print(
                    ' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))

                logrecord += ' Test site-{:d}| Test Acc: {:.4f}'.format(
                    client_idx, test_acc)
                test_acc_list[client_idx] = test_acc

    np_log = np.array(log, dtype=float)
    np.savetxt('./results/Global_Acc7.csv', np_log, delimiter=',', fmt='%.6f')
    np_log2 = np.array(log2, dtype=float)
    np.savetxt('./results/Global_bacc7.csv', np_log2, delimiter=',', fmt='%.6f')
