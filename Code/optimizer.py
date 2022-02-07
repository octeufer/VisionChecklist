from torch.optim import SGD, Adam

def initialize_optimizer(config, model):
    cfgopt = config['optimizer']
    if cfgopt =='SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(
            params,
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'])
    elif cfgopt == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(
            params,
            lr=config['lr'],
            weight_decay=config['weight_decay'])
    else:
        raise ValueError(f'Optimizer {cfgopt} not recognized.')

    return optimizer