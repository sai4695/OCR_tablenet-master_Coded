import argparse
import albumentations as album
import torch
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from tablenet import MarmotDataModule
from tablenet import TableNetModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_dir', type=str, default='./data/Marmot_data', help='Directory for the data')
    parser.add_argument('--weights_dir', type=str, default='./weights', help='Directory to save model weights')

    args = parser.parse_args()

    image_size = (896, 896)
    transforms_augmentation = album.Compose([
        album.Resize(1024, 1024, always_apply=True),
        album.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
        album.HorizontalFlip(),
        album.VerticalFlip(),
        album.Normalize(),
        ToTensorV2()
    ])

    transforms_preprocessing = album.Compose([
        album.Resize(*image_size, always_apply=True),
        album.Normalize(),
        ToTensorV2()
    ])

    complaint_dataset = MarmotDataModule(data_dir=args.data_dir, transforms_preprocessing=transforms_preprocessing,
                                         transforms_augmentation=transforms_augmentation, batch_size=2)

    model = TableNetModule(batch_norm=False)

    EXPERIMENT_NAME = f"{model.__class__.__name__}"
    logger = TensorBoardLogger('tb_logs', name=EXPERIMENT_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_top_k=5,
        save_last=True,
        mode="min",
        dirpath=args.weights_dir,
        filename='{epoch}-{validation_loss:.2f}'
    )

    early_stop_callback = EarlyStopping(monitor='validation_loss', mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], logger=logger, max_epochs=500,
                         gpus=1 if torch.cuda.is_available() else None)
    trainer.fit(model, datamodule=complaint_dataset)
    trainer.test()
