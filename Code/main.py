import os, csv
import sys
import datetime, time
import pandas as pd
import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

sys.path.append(os.getcwd())

from models.model import *
from train import *
from visualization import *
from scheduler import *
from optimizer import *
from loss import *
from load import *
from util import *

def main(cfgfile):
    configs = load_config(cfgfile)
    outdir = configs['outdir'][0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = outdir+'results_'+timestamp+'/'
    os.mkdir(outdir)
    out_file = '%s/results.txt' % outdir
    # output_file = '%s/output.all' % outdir
    cfg = sample_config(configs)
    
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    log(logfile, ("configs: %s" % str(cfg)))

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(logfile, ("device: %s" % device))

    # Random seed
    seed = cfg['seed']
    set_seed(seed)
    log(logfile, ("seed: %s" % str(seed)))

    # Transform
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor()
    ])
    log(logfile, ("processed imagesize: %d, %d" % (cfg['image_size'],cfg['image_size'])))

    # Data
    datapath = cfg['datadir'] + cfg['dataset']
    batchsize = cfg['batchsize']
    datalabel = cfg['datalabel']
    if datalabel == 'Tinyimagenet':
        dataset, datasets, dataloaders = load_Tinyimagenet(datapath, batchsize, transform, logfile)
    elif datalabel == 'Waterbirds':
        datasets, dataloaders = load_Waterbirds(datapath, batchsize, logfile)
    elif datalabel == 'CIFAR10':
        datasets, dataloaders = load_cifar10(datapath, batchsize, cfg['image_size'], logfile)
    elif datalabel == 'CIFAR100':
        datasets, dataloaders = load_cifar100(datapath, batchsize, cfg['image_size'], logfile)
    elif datalabel == 'Camelyon17':
        datasets, dataloaders = load_camelyon17(datapath, transform, batchsize, logfile)
    elif datalabel == 'CheXpert':
        datasets, dataloaders = load_chexpert(datapath, cfg['image_size'], batchsize, logfile)
    elif datalabel == 'Imagenet-A':
        datasets, dataloaders = load_imageneta(datapath, batchsize, transform, logfile)
    elif datalabel == 'Imagenet-R':
        datasets, dataloaders = load_imagenetr(datapath, batchsize, transform, logfile)
    elif datalabel == 'Imagenet-C':
        datasets, dataloaders = load_imagenetc(datapath, batchsize, transform, logfile)
    log(logfile, ("dataset loaded"))
    
    # Model
    model = initialize_model(cfg)
    is_trained = False
    model.to(device)
    log(logfile, ("model: %s loaded" % cfg['model']))

    # Initialize
    optimizer = initialize_optimizer(cfg, model)
    log(logfile, ("optimizer: %s initialized" % cfg['optimizer']))
    scheduler = initialize_scheduler(cfg, optimizer)
    log(logfile, ("scheduler: %s initialized" % cfg['scheduler']))
    loss = initialize_loss(cfg)
    log(logfile, ("loss function: %s initialized" % cfg['loss']))

    if cfg['train'] == True:
    # Train
        if cfg['resume'] == True:
            # Resume
            save_path = cfg['model_path']
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            best_acc = checkpoint['best val acc']
            cfg['n_epochs'] = epoch

        model, _ = train_model(model, dataloaders, loss, optimizer, scheduler, cfg['n_epochs'], device, outdir, logfile)
        is_trained = True
    # Evaluation
    if not is_trained and cfg['model_path'] != '':
        save_path = cfg['model_path']
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint['best val acc']
        log(logfile, ('best val acc: %s' % best_acc))
    
    if datalabel == 'Imagenet-A':
        test_model_imga(dataloaders['test'], model, device, loss, cfg, logfile)
    elif datalabel == 'Imagenet-R':
        test_model_imgr(dataloaders['test'], model, device, loss, cfg, logfile)
    elif datalabel == 'Imagenet-C':
        test_model_imgc(dataloaders, model, device, loss, cfg, logfile)
    else:
        test_model(dataloaders['test'], model, device, loss, cfg, logfile)
    # Visualization
    if cfg["visualization"] == True:
        # transform = transforms.Compose([
        # transforms.Resize((cfg['image_size'],cfg['image_size'])),
        # lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),   # tensor in [0,1]
        # transforms.Normalize(mean=[0.5330], std=[0.0349]),                                           # whiten with dataset mean and std
        # lambda x: x.expand(3,-1,-1)]) 

        # vis_data = ChexpertSmall(datapath, 'vis', transform)
        # vis_loader = DataLoader(vis_data, batchsize, shuffle=False, num_workers=0)

        if cfg['model'] == 'Resnet50':
            grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        elif cfg['model'] == 'vit_base_patch16_224':
            grad_cam_hooks = {'forward': model.blocks[-1].norm1, 'backward': model.head}
            
        visualize(model, dataloaders['vis'], grad_cam_hooks, device, outdir)

    f.close()
    

if __name__ == '__main__':
	main(sys.argv[1])
