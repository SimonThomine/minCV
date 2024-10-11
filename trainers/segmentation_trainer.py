import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from trainers.base_trainer import BaseTrainer
from minBackbones import BaseLayer,ViTParams
from dataset.load_data import load_seg_dataset



def iou_multiclass(pred, label, threshold=0.5):
    pred = (pred > threshold).astype(np.float32) # Binarisation des prédictions
    ious = []
    num_classes = pred.shape[0]  # Nombre de classes = nombre de canaux
    
    for cls in range(num_classes):
        pred_cls = pred[cls]  # Canal pour la classe actuelle
        label_cls = label[cls]  # Canal correspondant pour la vérité terrain
        
        intersection = (pred_cls * label_cls).sum()
        union = pred_cls.sum() + label_cls.sum() - intersection
        if union == 0:  # Eviter la division par zéro
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    
    return sum(ious) / num_classes

def dice_multiclass(pred, label, threshold=0.5):
    pred = (pred > threshold).astype(np.float32)  # Binarisation des prédictions
    dices = []
    num_classes = pred.shape[0]
    
    for cls in range(num_classes):
        pred_cls = pred[cls]
        label_cls = label[cls]
        
        intersection = (pred_cls * label_cls).sum()
        dice = (2.0 * intersection) / (pred_cls.sum() + label_cls.sum())
        dices.append(dice.item())
    
    return sum(dices) / num_classes  # Moyenne du Dice pour toutes les classes

def cal_iou(preds, labels, threshold=0.5):
    tot_iou = 0
    for pred,label in zip(preds,labels):
        tot_iou += iou_multiclass(pred, label, threshold)
    return tot_iou / len(preds)
      

# def cal_iou(preds, labels, threshold=0.5):
#     tot_iou = 0
#     for pred,label in zip(preds,labels):
#       pred = torch.argmax(pred, dim=1)  # Prédiction multiclasses
#       for cls in range(num_classes):
#         pred_cls = (preds == cls).float()
#         label_cls = (labels == cls).float()
#         ious.append(iou(pred_cls, label_cls, threshold))
#     return tot_iou / len(preds)

class SegTrainer(BaseTrainer):
  def __init__(self, data):
    super().__init__(data)
    
    assert "layers" in self.data and all(isinstance(x, BaseLayer) for x in self.data["layers"]), "layers not found in data or not a list of Layer objects"

    self.model_dir = f"models/{data['model_family']}_segmentation_{self.data['dataset']}"
    os.makedirs(self.model_dir, exist_ok=True)

    if self.classes>2:
      self.criterion=torch.nn.CrossEntropyLoss()
    else:
      self.criterion=torch.nn.BCEWithLogitsLoss()
    
  def load_model(self):
    super().load_model()
    self.model=self.backbone(image_dim=self.image_dim,classes=self.classes,**self.data).to(self.data["device"])

  def load_optim(self):
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.data["lr"])
    
  def change_mode(self,period="train"):
    if period=="train":
      self.model.train()
    else:
      self.model.eval()

  def load_data(self):
      
      self.train_loader,self.val_loader,self.test_loader,dataset_info=load_seg_dataset(**self.data)
      self.image_dim=dataset_info.image_dim
      self.classes=dataset_info.classes

      
  def infer(self):
      image,self.label=self.sample
      image=image.to(self.data["device"])
      self.label=self.label.to(self.data["device"]).float()
      self.pred=self.model(image)
    
  def compute_loss(self):
      return self.criterion(self.pred,self.label)
  
  def save_checkpoint(self):
      torch.save(self.model.state_dict(), self.model_dir+"/model.pth")
    
  def load_weights(self):
      self.model.load_state_dict(torch.load(self.model_dir+"/model.pth"))
    

  def cal_score(self):
      
      import matplotlib.pyplot as plt
      plt.imshow(self.pred[0].detach().cpu().numpy().argmax(0))
      plt.show()
      plt.imshow(self.label[0].detach().cpu().numpy().argmax(0))
      plt.show()
      
      return self.pred.cpu().numpy(),self.label.cpu().numpy()
  
  def compute_metrics(self,scores):
      all_predictions = []
      all_labels = []
    
      # Parcourir la liste des tuples (predictions, labels)
      for predicted, labels in scores:
          all_predictions.extend(predicted)
          all_labels.extend(labels)  
      


      iou= cal_iou(all_predictions, all_labels)

      # Afficher ou enregistrer les métriques
      print(f"IoU: {iou:.4f}")
      # print(f"Precision: {precision:.4f}")
      # print(f"Recall: {recall:.4f}")
      # print(f"F1 Score: {f1:.4f}")

      #return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}