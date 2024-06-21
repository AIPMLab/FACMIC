import torch
from utils.clip_util import AverageMeter
import utils.clip_util as clu
import torch.nn as nn
import clip
from utils.loss_function import CrossEntropyLabelSmooth
from utils.clip_util import convert_models_to_fp32
from utils.clip_util import FocalLossWithSmoothing
from tqdm import tqdm

def totrain(model):
    model.model.train()
    model.fea_attn.train()


def train(args, model, data_loader, optimizer, device, testloader, mmd_loss, server_model, previous_nets):
    totrain(model)
    texts = model.labels
    t_features = clu.get_text_features_list(texts, model.model).float() #update each round based on the model
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_loss_clf = AverageMeter()
    train_loss_transfer = AverageMeter()
    print(len(data_loader), len(testloader))
    i = 0
    loss_all = 0
    if args.method == 'ours':
        for batch, batch_t in zip(data_loader, testloader):
            i+=1
            # print(i)
            if i == args.n_iter: # use it for BT, SC dataset, except Real.
                break
            image, text, label = batch
            image_t, text_t, label_t = batch_t
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_t = image_t.to(device)
                image_features = model.model.encode_image(image).float()
                test_features = model.model.encode_image(image_t).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)

                i_features = clu.get_image_features(
                image_t, model.model, model.preprocess).float()
                i_attn = model.fea_attn(i_features)
                i_features = torch.mul(i_attn, i_features)
                # test_features = torch.mul(i_attn, test_features)
                similarity = clu.get_similarity(i_features, t_features)

                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)
                test_features = test_features / \
                test_features.norm(dim=1, keepdim=True)

                loss_m = mmd_loss(image_features, test_features, label, similarity)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2

                loss = cla_loss + loss_m

                train_loss_clf.update(cla_loss.item())
                train_loss_transfer.update(loss_m.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg, 'trans loss:', train_loss_transfer.avg)
    if args.method == 'fedprox':
        for batch in (data_loader):
            i += 1
            if i == args.n_iter: # use it for BT, SC dataset, except Real.
                break
            image, text, label = batch
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                # print(loss)
                # loss_all += loss
                if args.step > 0:
                    w_diff = torch.tensor(1e-10, device=device)
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2).float()  # model difference
                        # print(w_diff)
                    w_diff = torch.sqrt(w_diff)
                    train_loss_transfer.update((1e-2 / 2. * w_diff).item())
                    loss += 1e-2 / 2. * w_diff  # dif loss
                    # print(loss)
                optimizer.zero_grad()
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg, 'w_diff loss: ', train_loss_transfer.avg)
    if args.method == 'fedavg':
        for batch in (data_loader):
            i += 1
            if i == args.n_iter: # use it for BT, SC dataset, except Real.
                break
            image, text, label = batch
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)
                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
                train_loss_clf.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'moon':
        cnt = 0
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion = nn.CrossEntropyLoss()
        mu = 1
        for batch in data_loader:
            optimizer.zero_grad()
            i += 1
            if i == args.n_iter:
                break
            image, text, label = batch
            image = image.to(device)
            text = text.to(device)
            image_features = model.model.encode_image(image).float()
            image_features_glo = server_model.model.encode_image(image).float()
            text_features = model.model.encode_text(text).float()
            image_features = image_features / \
                             image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                            text_features.norm(dim=1, keepdim=True)
            image_features_glo = image_features_glo / \
                             image_features_glo.norm(dim=1, keepdim=True)
            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_image_glo = logit_scale * image_features_glo @ text_features_glo.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
            train_loss_clf.update(loss.item())
            # MOON contrastive loss below, we refered the original codes, it needs [logits_per_image] to measure.
            # Model-Contrastive Federated Learning
            posi = cos(image_features, image_features_glo)
            logits = posi.reshape(-1, 1)
            if args.step > 0:
                image_features_pre = previous_nets.model.encode_image(image).float()
                text_features_pre = previous_nets.model.encode_text(text).float()
                image_features_pre = image_features_pre / \
                                     image_features_pre.norm(dim=1, keepdim=True)
                nega = cos(image_features, image_features_pre)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= args.temp
                labels = torch.zeros(image.size(0)).cuda().long()
                loss += mu * criterion(logits, labels)
                train_loss_transfer.update(mu * criterion(logits, labels))
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg, 'MOON loss: ', train_loss_transfer.avg)
    if args.method == 'fedfocal':
        loss_img = FocalLossWithSmoothing(num_classes=32) # Batch size
        loss_txt = FocalLossWithSmoothing(num_classes=32)
        for batch in data_loader:
            optimizer.zero_grad()
            i += 1
            if i == args.n_iter:
                break
            image, text, label = batch
            image = image.to(device)
            text = text.to(device)
            image_features = model.model.encode_image(image).float()
            text_features = model.model.encode_text(text).float()
            image_features = image_features / \
                             image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                            text_features.norm(dim=1, keepdim=True)
            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth)) / 2
            train_loss_clf.update(loss.item())
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print("cla loss: ", train_loss_clf.avg)
    if args.method == 'fedclip':
        for batch in data_loader:
            i+=1
            # print(i)
            if i == args.n_iter: # use it for BT, SC dataset, except Real.
                break
            image, text, label = batch
            if len(text) > 1:
                image = image.to(device)
                text = text.to(device)
                image_features = model.model.encode_image(image).float()
                text_features = model.model.encode_text(text).float()
                image_features_att = model.fea_attn(image_features)
                image_features = torch.mul(image_features_att, image_features)


                image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

                cla_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/  2

                train_loss_clf.update(cla_loss.item())
                optimizer.zero_grad()
                cla_loss.backward()
                optimizer.step()
        print("cla loss: ", train_loss_clf.avg)
