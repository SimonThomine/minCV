import os
import torch
from utils.util import AverageMeter,create_scheduler
from tqdm import tqdm
from minBackbones import BACKBONES,BaseLayer,ViTParams,verify_model_family

class BaseTrainer:          
    def __init__(self, data):  
        
        self.data=data
        

        assert "lr" in self.data and isinstance(self.data["lr"],float), "lr not found in data or not a float"
        assert "batch_size" in self.data and isinstance(self.data["batch_size"],int), "batch_size not found in data or not an int"
        assert "num_epochs" in self.data and isinstance(self.data["num_epochs"],int), "num_epochs not found in data or not an int"
        assert "device" in self.data, "device not found in data"

        self.load_data()
        self.load_model()
        self.load_optim()
        
        create_scheduler(self)
    
    
    def load_optim(self):
        pass    
    
    def load_model(self):
        model_family=verify_model_family(self.data.get("layers",None),self.data.get("model_family",None))
        self.data["model_family"]=model_family
        self.backbone=BACKBONES[self.data["model_family"]]
        pass
    
    def change_mode(self,period="train"):
        pass
    
    def prepare_epoch(self):
        pass
    
    def prepare_batch(self):
        pass
    
    def infer(self):
        pass
    
    def compute_loss(self):
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
        epoch_bar = tqdm(total=len(self.train_loader) * self.data["num_epochs"],desc="Training",unit="batch")
        
        for _ in range(1, self.data["num_epochs"] + 1):
            
            self.prepare_epoch()
            
            losses = AverageMeter()
            for sample in self.train_loader:
                self.sample=sample
                
                self.prepare_batch()

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                  
                    self.infer()
                    loss=self.compute_loss()
                    loss.backward()
                    losses.update(loss.sum().item(), sample[0].size(0))
                    self.optimizer.step()
                    if self.scheduler_type=='OneCycleLR':
                        self.scheduler.step()
                    epoch_bar.update()
                    
            
            
            
            if self.scheduler_type in ['ReduceLROnPlateau','StepLR']:
                self.scheduler.step()            
            
            val_loss = self.val(epoch_bar)
            
            epoch_bar.set_postfix({"Train Loss": losses.avg, "Val Loss": val_loss})
            
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
        
        for sample in self.val_loader: 
            self.sample=sample            
            self.prepare_batch()
            
            with torch.set_grad_enabled(False):
                
                self.infer()
                loss=self.compute_loss()
                losses.update(loss.sum().item(), sample[0].size(0))
                
        epoch_bar.set_postfix({"Val Loss": loss.item()})

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
    
    