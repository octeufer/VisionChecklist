import torch
import timm
import torchvision
from models.vit import ViT
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel

def initialize_model(config):
    if config['model'] == 'ViT':
        model = ViT(
            image_size = config['image_size'],
            patch_size = config['patch_size'],
            num_classes = config['num_classes'],
            dim = config['dim'],
            depth = config['depth'],
            heads = config['heads'],
            mlp_dim = config['mlp_dim'],
            dropout = config['dropout'],
            emb_dropout = config['emb_dropout']
        )
    elif config['model'] == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config['num_classes'])
    elif config['model'] == 'ViT_pretrain':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    elif config['model'] == 'Resnet50':
        # model = torchvision.models.resnet50(pretrained=True)
        model = timm.create_model('resnet50', pretrained=True, num_classes=config['num_classes'])
    elif config['model'] == 'Deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    return model