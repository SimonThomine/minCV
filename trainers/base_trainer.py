import os
import torch
from utils.util import AverageMeter,create_scheduler
from tqdm import tqdm

class BaseTrainer:          
    def __init__(self, data):  
        
        
        self.data=data
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.load_model()
        
        self.load_data()
        
        self.load_optim()
        
        create_scheduler(self)
    
    
    def load_optim(self):
        pass    
    
    def load_model(self):
        pass
    
    def change_mode(self,period="train"):
        pass
    
    def prepare_epoch(self):
        pass
    
    def prepare_batch(self):
        pass
    
    def infer(self,image,test=False):
        pass
    
    def computeLoss(self):
        pass
    
    def cal_score(self):
        pass

    def save_checkpoint(self):
      pass
    
    def load_weights(self):
        pass
    
    def post_process(self):
        pass
    
    def load_data(self):
        pass

    def train(self):
        
        self.change_mode("train")
        
        best_score = None
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        
        for _ in range(1, self.num_epochs + 1):
            
            self.prepare_epoch()
            
            for sample in self.train_loader:
                self.sample=sample
                
                self.prepare_batch()

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                  
                    self.infer()
                    loss=self.computeLoss()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler_type=='OneCycleLR':
                        self.scheduler.step()
                    
                    
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()
            
            if self.scheduler_type in ['ReduceLROnPlateau','StepLR']:
                self.scheduler.step()            
            
            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

        epoch_bar.close()
        
        print("Training end.")

    def val(self, epoch_bar):
        self.change_mode("eval")
        losses = AverageMeter()
        
        self.load_iter()
        
        for sample in self.val_loader: 
            self.sample=sample            
            self.prepare_batch()
            
            with torch.set_grad_enabled(False):
                
                self.infer()
                loss=self.computeLoss()
                
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    
    @torch.no_grad()
    def test(self):

        self.load_weights()
        self.change_mode("eval")

        scores = []
        progressBar = tqdm(self.test_loader)
        for sample in self.test_loader:
            self.sample=sample
            with torch.set_grad_enabled(False):
                
                self.infer() 
                  
                self.post_process()
                
                score=self.cal_score()
                
                progressBar.update()
                
            scores.append(score)

        progressBar.close()
        
        self.compute_metrics(scores)
    
    