from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR, MultiStepLR

def initialize_scheduler(config, optimizer):
    sch = config['scheduler']
   
    if sch is None:
        return None
    elif sch == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer)
    elif sch == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif sch == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30)
    elif sch == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=[30,80])
    
    return scheduler
