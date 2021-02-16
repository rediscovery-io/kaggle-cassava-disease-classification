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
from torchvision import transforms

# Python Imports
import numpy as np
from PIL import Image
import glob
import random
import pandas as pd
random.seed(123)
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

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

train = pd.read_csv('../../remo_train.csv')
valid = pd.read_csv('../../remo_valid.csv')


train_transforms = SwAVTrainDataTransform(size_crops=[512], min_scale_crops=[0.33], max_scale_crops=[0.33], nmb_crops=[0], normalize=imagenet_normalization())
val_transforms = SwAVEvalDataTransform(size_crops=[512], min_scale_crops=[0.33], max_scale_crops=[0.33], nmb_crops=[0], normalize=imagenet_normalization())

train_dl = DataLoader(CustomDataset(data_path=train, transforms=train_transforms), shuffle=True, batch_size=128, num_workers = 6, pin_memory=True)

val_dl = DataLoader(CustomDataset(data_path=valid, transforms=val_transforms), batch_size=64, num_workers = 6, pin_memory=True)

from pl_bolts.models.self_supervised import SwAV

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

#swav.freeze()
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
        self.log('Training Loss', loss)
        self.log('Training Accuracy', accuracy(preds, y))

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        with torch.no_grad():
            features = self.backbone(x)[-1]
        
        out = self.finetune_layer(features)
        
        val_loss = cross_entropy(out, y)
        val_acc = accuracy(out, y)
        
        self.log('Validation Loss', val_loss)
        self.log('Validation Accuracy', val_acc)
        
        return val_loss
        

    def forward(self, x):
      with torch.no_grad():
        feats = self.backbone(x)[-1]
        out = self.finetune_layer(feats)
      return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
        return [optimizer]
    
trainer = pl.Trainer(
        gpus = 2,
        max_epochs = 20,
        accelerator= 'ddp',
        plugins='ddp_sharded'
    )
classifier = ImageClassifier()

trainer.fit(classifier, train_dl, val_dl)




