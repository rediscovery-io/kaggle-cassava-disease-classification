""" Self-supervised transfer learning for Cassava Dataset Classification """

# Lightning Imports
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
from pytorch_lightning.metrics.functional import accuracy

# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torchvision as tv

# Python Imports
import numpy as np
import pandas as pd
from PIL import Image
import glob
import random
random.seed(123)

class CustomDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        im = Image.open('train_images/'+self.data_path.loc[idx, 'image_id'])

        label = self.data_path.loc[idx, 'label']


        if self.transforms:
            im = self.transforms(im)

        return im, label

train_data = pd.read_csv('train.csv')

means =  [0.485, 0.456, 0.406]
stds  =  [0.229, 0.224, 0.225]


tv_transforms  =  tv.transforms.Compose([tv.transforms.Resize((500, 500), Image.BILINEAR), tv.transforms.ToTensor()])


train_dl = DataLoader(CustomDataset(data_path=train_data, transforms=tv_transforms), batch_size=128, num_workers = 4, pin_memory=True)

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes = 5, lr = 1e-3):
        super().__init__()
        self.backbone = swav.model
        self.num_classes = num_classes
        
        self.finetune_layer = nn.Sequential(nn.Linear(3000, 256),
                         nn.ReLU(),
                         nn.Dropout(p=0.5),
                         nn.Linear(256, self.num_classes),
                         nn.LogSoftmax(dim=1))

    def training_step(self, batch, batch_idx):

        x, y = batch

        with torch.no_grad():
            features = self.backbone(x)[-1]

        preds = self.finetune_layer(features)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', accuracy(preds, y))

        return loss

    def forward(self, x):
      with torch.no_grad():
        feats = self.backbone(x)[-1]
        out = self.finetune_layer(feats)
      return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
        return [optimizer], [scheduler]
    
weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

classifier = ImageClassifier(num_classes=5)

trainer = pl.Trainer(gpus = 2, max_epochs=30, accelerator='ddp', plugins='ddp_sharded',
                    resume_from_checkpoint='lightning_logs/version_2/checkpoints/epoch=14-step=1259.ckpt')
trainer.fit(classifier, train_dl)



