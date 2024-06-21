

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset =='BrainTumor':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT4':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT2':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT_iid':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT_Large':
        domains = ['client_0', 'client_1', 'client_2', 'client_3','client_4','client_5']
    if dataset =='Alzheimer':
        domains = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5','client_6', 'client_7', 'client_8', 'client_9']
    if dataset =='Skin':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='Skin2':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='Skin_Large':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='Skin4':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='RealSkin':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='SkinCen':
        domains = ['client_0', 'client_1']
    args.domains = domains
    if args.dataset =='BrainTumor':
        args.num_classes = 4
    if args.dataset =='BT4':
        args.num_classes = 4
    if args.dataset =='BT2':
        args.num_classes = 4
    if args.dataset =='BT_iid':
        args.num_classes = 4
    if args.dataset =='BT_Large':
        args.num_classes = 4
    if args.dataset =='Alzheimer':
        args.num_classes = 4
    if args.dataset =='Skin':
        args.num_classes = 9
    if args.dataset =='Skin2':
        args.num_classes = 9
    if args.dataset =='Skin_Large':
        args.num_classes = 6
    if args.dataset =='Skin4':
        args.num_classes = 9
    if args.dataset =='BT44':
        args.num_classes = 14
    if args.dataset =='RealSkin':
        args.num_classes = 7
    if args.dataset =='SkinCen':
        args.num_classes = 7
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
