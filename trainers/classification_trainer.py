import os
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from trainers.base_trainer import BaseTrainer
from minBackbones.mlp import Mlp
from minBackbones.layers import BaseLayer,MlpLayer,CnnLayer
from dataset.load_data import load_classi_dataset

class ClassiTrainer(BaseTrainer):
  def __init__(self, data):
    super().__init__(data)
    
    assert "layers" in self.data and all(isinstance(x, BaseLayer) for x in self.data["layers"]), "layers not found in data or not a list of Layer objects"
    assert "lr" in self.data and isinstance(self.data["lr"],float), "lr not found in data or not a float"
    
    self.model_dir = f"models/{data['model_family']}_classification_{self.data['dataset']}"
    os.makedirs(self.model_dir, exist_ok=True)
    if self.classes>2:
      self.criterion=torch.nn.CrossEntropyLoss()
    else:
      self.criterion=torch.nn.BCEWithLogitsLoss()
    
  def load_model(self):
    # Pas forcément mlp, peut être cnn, transformers, rnn, etc
    self.model=Mlp(classes=self.classes,input_dim=self.input_dim,**self.data).to(self.data["device"])
    
  def load_optim(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.data["lr"])
    
  def change_mode(self,period="train"):
    if period=="train":
      self.model.train()
    else:
      self.model.eval()

  def load_data(self):
      print(self.data["image_size"])
      self.train_loader,self.val_loader,self.test_loader,self.classes,self.input_dim=load_classi_dataset(**self.data)
      
  def infer(self):
      image,self.label=self.sample
      image=image.to(self.data["device"])
      self.label=self.label.to(self.data["device"])
      self.pred=self.model(image)
    
  def compute_loss(self):
      return self.criterion(self.pred,self.label)
  
  def save_checkpoint(self):
      torch.save(self.model.state_dict(), self.model_dir+"/mlp.pth")
    
  def load_weights(self):
      self.model.load_state_dict(torch.load(self.model_dir+"/mlp.pth"))
    
  def cal_score(self):
      print(self.pred.shape)
      if self.classes>2:
        _, prediction = torch.max(self.pred, 1)
      else:
        prediction = torch.round(torch.sigmoid(self.pred))

      return prediction.cpu().numpy(),self.label.cpu().numpy()
  
  def compute_metrics(self,scores):
      all_predictions = []
      all_labels = []
    
      # Parcourir la liste des tuples (predictions, labels)
      for predicted, labels in scores:
          all_predictions.extend(predicted)
          all_labels.extend(labels)  
      precision = precision_score(all_labels, all_predictions, average='macro')
      recall = recall_score(all_labels, all_predictions, average='macro')
      f1 = f1_score(all_labels, all_predictions, average='macro')
      accuracy = accuracy_score(all_labels, all_predictions)

      # Afficher ou enregistrer les métriques
      print(f"Accuracy: {accuracy:.4f}")
      print(f"Precision: {precision:.4f}")
      print(f"Recall: {recall:.4f}")
      print(f"F1 Score: {f1:.4f}")

      return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}