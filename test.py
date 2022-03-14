import monai
import os
import glob
import logging
import torch
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
)

keys = ("image", "label")

xforms = [
    LoadImaged(keys)
]

data_folder = "C:\\data\\COVID-19-20_v2\\Train"

images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
train_files = [{"image": img, "label": seg} for img, seg in zip(images, labels)]
train_set = monai.data.CacheDataset(train_files, transform=xforms)
train_loader = monai.data.DataLoader(train_set)
num_classes = 2
net = monai.networks.nets.BasicUNet(spatial_dims=3, in_channels=1, out_channels=2)
criterion = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = monai.engines.SupervisedTrainer(device=device, max_epochs=5, train_data_loader=train_loader, network=net, optimizer=optimizer, loss_function=criterion)
trainer.run()
