import torch
from utils.clip_util import AverageMeter
import utils.clip_util as clu
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def toeval(model):
    model.model.eval()
    model.fea_attn.eval()


def test(args, model, data_loader, device):
    toeval(model)
    total = 0
    correct = 0
    bacc = 0
    Prediction = []
    Label = []
    texts = model.labels
    text_features = clu.get_text_features_list(texts, model.model).float()
    if args.method == 'ours' or args.method == 'fedclip':
        with torch.no_grad():
            for batch in tqdm(data_loader):
                image, _, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                # print(image_features_attn)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()
                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
                # print(np.array(res)[:, 1],np.array(res)[:, 0])
            all_preds = np.concatenate(Prediction)
            all_labels = np.concatenate(Label)
            bacc = balanced_accuracy_score(all_labels, all_preds)
    else:
        bacc = 0
        Prediction = []
        Label = []
        with torch.no_grad():
            for batch in data_loader:
                image, _, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_labels = np.concatenate(Label)
            bacc = balanced_accuracy_score(all_labels, all_preds)
    return correct/total, bacc
