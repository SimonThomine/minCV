import os
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from trainers.base_trainer import BaseTrainer
from minBackbones.mlp import Mlp
from dataset.load_data import load_autoencoder_dataset
from minBackbones.layers import BaseLayer


class AeTrainer(BaseTrainer):
  def __init__(self, data):
    super().__init__(data)
    
    assert "layers" in self.data and all(isinstance(x, BaseLayer) for x in self.data["layers"]), "layers not found in data or not a list of Layer objects"
    assert "lr" in self.data and isinstance(self.data["lr"],float), "lr not found in data or not a float"
    
    self.model_dir = f"models/{data['model_family']}_autoencoder_{self.data['dataset']}"
    os.makedirs(self.model_dir, exist_ok=True)
    
    self.criterion=torch.nn.MSELoss()
    
  def load_model(self):
    # Pas forcément mlp, peut être cnn, transformers, rnn, etc
    self.model=Mlp(input_dim=self.input_dim,**self.data).to(self.data["device"])
    
  def load_optim(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.data["lr"])
    
  def change_mode(self,period="train"):
    if period=="train":
      self.model.train()
    else:
      self.model.eval()

  def load_data(self):
      print(self.data["image_size"])
      self.train_loader,self.val_loader,self.test_loader,self.input_dim=load_autoencoder_dataset(**self.data)
      
  def infer(self):
      self.image,self.label=self.sample
      self.image=self.image.to(self.data["device"])
      self.pred=self.model(self.image)
    
  def compute_loss(self):
      self.pred=self.pred.view(self.image.size())
      return self.criterion(self.pred,self.image)
  
  def save_checkpoint(self):
      torch.save(self.model.state_dict(), self.model_dir+"/mlp.pth")
    
  def load_weights(self):
      self.model.load_state_dict(torch.load(self.model_dir+"/mlp.pth"))
    
  def cal_score(self):
      self.pred=self.pred.view(self.image.size())
      mse_pixel_score = nn.MSELoss(reduction='none')
      per_pixel_score = mse_pixel_score(self.pred, self.image)
      score = per_pixel_score.view(per_pixel_score.size(0), -1).mean(dim=1).cpu().numpy()
      return score,self.label
  
  def compute_metrics(self,scores):
      all_dists = []
      all_labels = []
    
      # Parcourir la liste des tuples (predictions, labels) sur les batch de test
      for dists, labels in scores:
          all_dists.extend(dists)
          all_labels.extend(labels)  
      auroc=roc_auc_score(all_labels, all_dists)
      print(f"AUROC : {auroc:.4f}")

      return {"auroc": auroc}