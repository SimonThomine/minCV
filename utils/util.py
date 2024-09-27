import torch
import math
import torch.optim as optim

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def create_scheduler(trainer,scheduler_type='OneCycleLR'):
  trainer.scheduler_type=scheduler_type
  lr=trainer.data['lr']
  num_epochs=trainer.data['num_epochs']
  if scheduler_type=='OneCycleLR':
    trainer.scheduler=optim.lr_scheduler.OneCycleLR(trainer.optimizer,max_lr=lr*10,epochs=num_epochs,steps_per_epoch=len(trainer.train_loader))
  elif scheduler_type=='ReduceLROnPlateau':
    trainer.scheduler=optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
  elif scheduler_type=='StepLR':
    trainer.scheduler=optim.lr_scheduler.StepLR(trainer.optimizer,step_size=10)
  elif scheduler_type:
    raise Exception("Invalid scheduler type")