import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm
from PIL import Image
import multiprocessing
import pytorch_lightning as pl


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


means =  [0.485, 0.456, 0.406]
stds  =  [0.229, 0.224, 0.225]

train_transforms      =  transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(256),
                                             transforms.ToTensor()
                                            ])


val_transforms        =  transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(256),
                                             transforms.ToTensor(),
                                             transforms.Normalize(means, stds)])

train_data = pd.read_csv('../../remo_train.csv')
validation_data = pd.read_csv('../../remo_valid.csv')

batch_size = 32
epochs = 100
lr = 1e-3
n_gpus = 1
img_size = 256

num_workers = multiprocessing.cpu_count()

train_dl = DataLoader(CustomDataset(data_path=train_data, transforms=train_transforms), shuffle=True, batch_size=50, num_workers=num_workers, pin_memory=True)
#val_dl = DataLoader(CustomDataset(data_path=validation_data, transforms=val_transforms), batch_size = 50, num_workers=2, pin_memory=True)

class BYOLtrainer(pl.LightningModule):
  def __init__(self, net, **kwargs):
      super().__init__()
      self.learner = BYOL(net, **kwargs)

  def forward(self, images):
    return self.learner(images)

  def training_step(self, images, _):
    x, y = images
    loss = self.forward(x)
    self.log("loss", loss.item())

    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=lr)

resnet = models.resnet50(pretrained=True)

model = BYOLtrainer(
                resnet,
                image_size = 256,
                hidden_layer = 'avgpool',
                projection_size = 256,
                projection_hidden_size = 4096,
                moving_average_decay = 0.99)
                                                           

trainer = pl.Trainer(
        gpus = 2,
        max_epochs = 100,
        accumulate_grad_batches = 1,
        sync_batchnorm = True,
        accelerator= 'ddp',
        plugins='ddp_sharded'
    )

trainer.fit(model, train_dl)







