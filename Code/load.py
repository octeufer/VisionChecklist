import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

import torch
import torchvision

import torch.nn.functional as F

from torch.utils.data import DataLoader
from helper import *

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def log(logfile,str_in):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str_in+'\n')
    print(str_in)

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    #test
    # for v in cfg_sample.values():
    # 	print(type(v))
    return cfg_sample

def load_data(dataform, feature, dagform, logfile):
    log(logfile, ("data: %s" % dataform))
    log(logfile, ("feature: %s" % feature))
    log(logfile, ("DAG: %s" % dagform))

    df = pd.read_csv(dataform, sep=',', encoding='utf-8')

    dag = pd.read_csv(dagform, sep=';', encoding='utf-8')

    feature_out = None

    # if feature=='synthetic':
    #     feature_out = synthetic
    # elif feature=='colormnist':
    #     feature_out = colormnist
    # elif feature=='waterbirds':
    #     feature_out = waterbirds
    # elif feature=='hastie':
    #     feature_out =hastie

    return df, feature_out, dag
    # return feature_out, dag


def load_Tinyimagenet(datapath, batch_size, transform, logfile):
    log(logfile, ("data: %s" % datapath))

    dataset = torchvision.datasets.ImageFolder(datapath, transform=transform)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    datasets = dict()
    dataloaders = dict()

    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader

    return dataset, datasets, dataloaders

def load_imageneta(datapath, batch_size, transform, logfile):
    log(logfile, ("data: %s" % datapath))

    test_data = torchvision.datasets.ImageFolder(datapath, transform=transform)
    # train_data, val_data, test_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    datasets = dict()
    dataloaders = dict()

    # datasets['train'] = train_data
    # datasets['val'] = val_data
    datasets['test'] = test_data
    # dataloaders['train'] = train_loader
    # dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader

    return datasets, dataloaders

def load_imagenetr(datapath, batch_size, transform, logfile):
    log(logfile, ("data: %s" % datapath))

    test_data = torchvision.datasets.ImageFolder(datapath, transform=transform)
    # train_data, val_data, test_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    datasets = dict()
    dataloaders = dict()

    # datasets['train'] = train_data
    # datasets['val'] = val_data
    datasets['test'] = test_data
    # dataloaders['train'] = train_loader
    # dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader

    return datasets, dataloaders

def load_imagenetc(datapath, batch_size, transform, logfile):
    datasets = dict()
    dataloaders = dict()

    for c in CORRUPTIONS:
        log(logfile, ("corruption type: %s" % c))
        val_list = []
        for s in range(1, 6):
            valdir = os.path.join(datapath, c, str(s))
            valdata = torchvision.datasets.ImageFolder(valdir, transform=transform)
            val_loader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=0)
            val_list.append(val_loader)
        dataloaders[c] = val_list
    return datasets, dataloaders

def load_Waterbirds(datapath, batch_size, logfile):
    wbdata = WaterbirdsDataset(datapath)
    splits = ['train', 'val', 'test']
    subsets = wbdata.get_splits(splits)
    subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=wbdata.n_groups,
                              n_classes=wbdata.n_classes, group_str_fn=wbdata.group_str) \
                   for split in splits]
    
    log(logfile, ("preparing data"))
    train_data, val_data, test_data = subsets

    log(logfile, ("Setting up loader"))
    loader_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=None, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    datasets = dict()
    dataloaders = dict()

    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader
    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data

    return datasets, dataloaders

    
def load_cifar100(datapath, batchsize, imagesize, logfile, validation_split=0.1, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    log(logfile, ("preparing data"))
    transform_train = transforms.Compose([
        transforms.Resize((imagesize,imagesize)),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((imagesize,imagesize)),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    datasets = dict()
    dataloaders = dict()

    log(logfile, ("Setting up loader"))
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                             transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

    dataloaders['train'] = train_loader
    dataloaders['val'] = test_loader
    dataloaders['test'] = test_loader
    datasets['train'] = trainset
    datasets['val'] = testset
    datasets['test'] = testset

    return datasets, dataloaders

def load_cifar10(datapath, batchsize, imagesize, logfile, validation_split=0.1, max_threads=10):
    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
            transforms.Resize((imagesize,imagesize)),
            transforms.ToTensor(),
        #     transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        #                                             (4,4,4,4),mode='reflect').squeeze()),
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((imagesize,imagesize)),
        transforms.ToTensor(),
         normalize
    ])

    log(logfile, ("preparing data"))
    full_dataset = torchvision.datasets.CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10('_dataset', False, test_transform, download=False)


    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1

    log(logfile, ("Setting up loader"))
    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batchsize,
            num_workers=train_threads,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batchsize,
            num_workers=val_threads,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batchsize,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    datasets = dict()
    dataloaders = dict()

    dataloaders['train'] = train_loader
    dataloaders['val'] = valid_loader
    dataloaders['test'] = test_loader
    datasets['train'] = train_dataset
    datasets['val'] = val_dataset
    datasets['test'] = test_dataset

    return datasets, dataloaders

def load_camelyon17(datapath, transform, batchsize, logfile):
    dataset = get_dataset(dataset='camelyon17', root_dir=datapath)

    log(logfile, ("loading camelyon17 data"))

    train_data = dataset.get_subset('train', transform=transform)
    idval_data = dataset.get_subset('id_val', transform=transform)
    odval_data = dataset.get_subset('val', transform=transform)
    test_data = dataset.get_subset('test', transform=transform)

    log(logfile, ("building camelyon17 loader"))

    train_loader = get_train_loader('standard', train_data, batch_size=batchsize)
    idval_loader = get_eval_loader('standard', idval_data, batch_size=batchsize)
    odval_loader = get_eval_loader('standard', odval_data, batch_size=batchsize)
    test_loader = get_eval_loader('standard', test_data, batch_size=batchsize)

    datasets = dict()
    dataloaders = dict()

    datasets['train'] = train_data
    datasets['val'] = idval_data
    datasets['odval'] = odval_data
    datasets['test'] = test_data

    dataloaders['train'] = train_loader
    dataloaders['val'] = idval_loader
    dataloaders['odval'] = odval_loader
    dataloaders['test'] = test_loader

    return datasets, dataloaders

def load_chexpert(datapath, imagesize, batchsize, logfile):
    
    transform = transforms.Compose([
        transforms.Resize((imagesize,imagesize)),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        transforms.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
        lambda x: x.expand(3,-1,-1)]) 

    train_data = ChexpertSmall(datapath, 'train', transform)
    val_data = ChexpertSmall(datapath, 'valid', transform)
    test_data = ChexpertSmall(datapath + '/valid.csv', 'test', transform)
    vis_data = ChexpertSmall(datapath, 'vis', transform)

    train_loader = DataLoader(train_data, batchsize, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batchsize, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batchsize, shuffle=False, num_workers=0)
    vis_loader = DataLoader(vis_data, batchsize, shuffle=False, num_workers=0)

    datasets = dict()
    dataloaders = dict()

    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data

    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader
    dataloaders['vis'] = vis_loader

    return datasets, dataloaders