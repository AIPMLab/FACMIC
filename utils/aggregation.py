import torch


def communication(args, server_model, models, client_weights):
    client_num = len(models)
    if args.aggmode == 'att':
        with torch.no_grad():
            for key in server_model.fea_attn.state_dict().keys():
                if 'num_batches_tracked' in key or 'bert' in key:
                    server_model.fea_attn.state_dict()[key].data.copy_(
                    models[3].fea_attn.state_dict()[key]) # model[3] means the index of global model, depending on your settings.
                else:
                    temp = torch.zeros_like(server_model.fea_attn.state_dict()[
                                        key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            temp += client_weights[client_idx] * \
                                models[client_idx].fea_attn.state_dict()[key]
                    server_model.fea_attn.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            models[client_idx].fea_attn.state_dict()[key].data.copy_(
                            server_model.fea_attn.state_dict()[key])

    if args.aggmode == 'avg':
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key or 'bert' in key:
                    server_model.state_dict()[key].data.copy_(
                        models[1].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[
                                                key], dtype=torch.float32)
                    for client_idx in range(client_num):  # aggregation
                        if client_idx not in args.test_envs:
                            temp += client_weights[client_idx] * \
                                    models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):  # broadcast
                        if client_idx not in args.test_envs:
                            models[client_idx].state_dict()[key].data.copy_(
                                server_model.state_dict()[key])
    return server_model, models
