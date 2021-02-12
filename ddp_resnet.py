# PyTorch Lightning Imports
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torch.optim import Adam, AdamW
import torchvision.models as models


# Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default Python
import os
from PIL import Image

cat_to_idx = {'Cassava bacterial blight (cbb)': 0,
              'Cassava brown streak disease (cbsd)': 1,
              'Cassava green mottle (cgm)': 2,
              'Cassava mosaic disease (cmd)': 3,
              'Healthy': 4}


class CustomDataset(Dataset):
    def __init__(self, data_path, transforms, mapping = cat_to_idx):
        self.data_path = data_path
        self.transforms = transforms
        self.mapping = cat_to_idx

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        im = Image.open(self.data_path.loc[idx, 'file_name'])
        label = int(self.mapping[self.data_path.loc[idx, 'classes']])

        if self.transforms:
            im = self.transforms(im)
        return im, label

train_data = pd.read_csv('../remo_train.csv')
validation_data = pd.read_csv('../remo_valid.csv')

means =  [0.485, 0.456, 0.406]
stds  =  [0.229, 0.224, 0.225]


tv_transforms      =  tv.transforms.Compose([
                        tv.transforms.RandomRotation(30),
                         tv.transforms.RandomResizedCrop(400),
                         tv.transforms.RandomHorizontalFlip(p=0.5),
                         tv.transforms.ToTensor(),
                         tv.transforms.Normalize(means, stds)])

val_transforms = tv.transforms.Compose([ tv.transforms.Resize((400, 400), Image.BICUBIC),
                                         tv.transforms.ToTensor(),
                                         tv.transforms.Normalize(means, stds)])


                                
train_dl = DataLoader(CustomDataset(data_path=train_data, transforms=tv_transforms), batch_size=128, num_workers=4, pin_memory=True)
val_dl = DataLoader(CustomDataset(data_path=validation_data, transforms=val_transforms), batch_size = 10, num_workers=4, pin_memory=True)



class ResNetTransferLearning(pl.LightningModule):
  def __init__(self, num_classes, model):
      super().__init__()
      self.num_classes = num_classes
      self.backbone = model

      for param in self.backbone.parameters():
          param.required_grad = False

      self.backbone.fc = nn.Sequential(nn.Linear(512, 256), 
                                       nn.ReLU(), 
                                       nn.Dropout(p=0.5), 
                                       nn.Linear(256, self.num_classes), 
                                       nn.LogSoftmax(dim=1))

    
  def training_step(self, batch, batch_idx):
      x, y = batch
      preds = self.backbone(x)

      loss = cross_entropy(preds, y)
      self.log('train_loss', loss)
      self.log('train_acc', accuracy(preds, y))

    return loss

  def validation_step(self, batch, batch_idx):
      x, y = batch
      preds = self.backbone(x)
      loss = cross_entropy(preds, y)
      self.log('valid_loss', loss)
      self.log('valid_acc', accuracy(preds, y))
     

  def forward(self, x):
      with torch.no_grad():
          out = self.backbone(x)
      
      return out

  def configure_optimizers(self):
      optimizer = AdamW(self.parameters(), lr = 0.001)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5)
      return [optimizer], [scheduler]


num_classes = 5
model = models.resnet18(pretrained=True)
#model = models.resnet34(pretrained=True)
resnet_model = ResNetTransferLearning(num_classes= num_classes,
                                      model = model)


trainer = pl.Trainer(max_epochs=128, gpus=2, flush_logs_every_n_steps = 100, accelerator='ddp', plugins='ddp_sharded', resume_from_checkpoint='lightning_logs/version_7/checkpoints/epoch=107-step=7235.ckpt', check_val_every_n_epoch=5)
trainer.fit(resnet_model, train_dl, val_dl)
