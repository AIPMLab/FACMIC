import clip
import torchvision.datasets as datasets
from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):


    def __init__(self, dataset, root, preprocess, sign, prompt='a picture of a'):
        dataset = os.path.join(root, dataset)
        if sign:
            data = datasets.ImageFolder(dataset, transform=self._transform)
        else:
            data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]

        self.preprocess = preprocess
        self.text = clip.tokenize(self.labels)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _TRANSFORM = transforms.Compose([ #test
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    _transform = transforms.Compose( #train
        [transforms.Resize([224, 224]),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 91.6667
                              std=[0.229, 0.224, 0.225])])

def get_data(data_name):
    datalist = {'BrainTumor': 'BrainTumor', 'BT4': 'BT4', 'BT2': 'BT2', 'BT_iid': 'BT_iid', 'BT_Large': 'BT_Large', 'Alzheimer': 'Alzheimer', 'Skin': 'Skin', 'Skin2': 'Skin2', 'Skin_Large': 'Skin_Large', 'Skin4': 'Skin4', 'BT44': 'BT44', 'RealSkin': 'RealSkin', 'SkinCen': 'SkinCen'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def getfeadataloader(args, model):
    trl, val, tel, telt, tr_t = [], [], [], [], []
    trd, vad, ted, teld, tr_td = [], [], [], [], []
    for i, item in enumerate(args.domains):
        if i in args.test_envs: #target domain data
            data = ImageTextData(
                item, args.root_dir+args.dataset+'/', model.preprocess, sign=0)
            data2 = ImageTextData(
                item, args.root_dir + args.dataset + '/', model.preprocess, sign=1)
            model.setselflabel(data.labels)
            ted.append(torch.utils.data.DataLoader(
                data, batch_size=args.batch, shuffle=False))
            temp = get_data_loader(
                data2, args.batch, infinite_data_loader=True)
            telt.append(temp)
            trd.append(0)
            vad.append(0)
        else:
            data = ImageTextData(
                item, args.root_dir+args.dataset+'/', model.preprocess, sign=1)
            l = len(data)
            index = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(index)
            l1, l2, l3 = int(l*0.8), int(l*0.1), int(l*0.1)
            trl.append(torch.utils.data.Subset(data, index[:l1]))
            val.append(torch.utils.data.Subset(data, index[l1:l1+l2]))
            tel.append(torch.utils.data.Subset(data, index[l1+l2:l1+l2+l3]))
            tr_t.append(torch.utils.data.Subset(data, index[:l1]))
            # trd.append(torch.utils.data.DataLoader(
            #     trl[-1], batch_size=args.batch, shuffle=True, drop_last=True))
            trd.append(get_data_loader(trl[-1], batch_size=args.batch, shuffle=True, drop_last=True, infinite_data_loader=True))
            tr_td.append(get_data_loader(tr_t[-1], batch_size=args.batch, shuffle=True, drop_last=True,
                                       infinite_data_loader=False))
            vad.append(torch.utils.data.DataLoader(
                val[-1], batch_size=args.batch, shuffle=False,drop_last=True))
            ted.append(torch.utils.data.DataLoader(
                tel[-1], batch_size=args.batch, shuffle=False,drop_last=False))
    return trd, vad, ted, telt, tr_td

def BrainTumor(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def BT4(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def BT2(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def BT_iid(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def BT_Large(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def Alzheimer(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def Skin(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def Skin2(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def Skin_Large(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def Skin4(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def BT44(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def RealSkin(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def SkinCen(args, model):
    trd, vad, ted, telt, tr_td = getfeadataloader(args, model)
    return trd, vad, ted, telt, tr_td

def get_data_loader(dataset, batch_size, shuffle=True, drop_last=True, infinite_data_loader=False,
                    **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                            **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                   **kwargs)


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=False,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=False)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0  # Always return 0
