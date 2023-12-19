import argparse
import albumentations as album
import torch
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import optuna

from tablenet import AuditImageDataModule
from tablenet import TableNetModule

# Objective function for Optuna


def objective(trial):
    """ Hyperparameters to be tuned by Optuna """
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # data module and model with suggested hyperparameters
    complaint_dataset_test = AuditImageDataModule(
        data_dir=args.data_dir,
        transforms_preprocessing=transforms_preprocessing,
        transforms_augmentation=transforms_augmentation,
        batch_size=batch_size
    )
    model_test = TableNetModule(learning_rate=learning_rate)

    # Set up the PyTorch Lightning trainer
    trainer_test = pl.Trainer(
        logger=False,  # Disable logging for hyperparameter optimization
        max_epochs=10,  # I haven't made this as a hyper parameter due to high computational cost involved.
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        callbacks=[EarlyStopping(monitor='validation_loss', patience=3)]
    )

    # Train the model
    trainer_test.fit(model_test, datamodule=complaint_dataset_test)

    # Return the performance metric to be optimized
    validation_loss = trainer_test.callback_metrics['validation_loss'].item()
    return validation_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for TableNet')
    parser.add_argument('--data_dir', type=str, default='./data/Marmot_data', help='Directory for the data')
    parser.add_argument('--weights_dir', type=str, default='./weights', help='Directory to save model weights')
    args = parser.parse_args()

    # Define the transformations
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

    # Run the Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # This can be specified

    # best hyperparameters
    best_hyperparams = study.best_trial.params
    print(f"Best hyperparameters: {best_hyperparams}")

    # best hyperparameters from the study
    best_batch_size = best_hyperparams['batch_size']
    best_learning_rate = best_hyperparams['learning_rate']

    # data module and model with the best hyperparameters
    complaint_dataset = AuditImageDataModule(
        data_dir=args.data_dir,
        transforms_preprocessing=transforms_preprocessing,
        transforms_augmentation=transforms_augmentation,
        batch_size=best_batch_size
    )
    model = TableNetModule(batch_norm=False, learning_rate=best_learning_rate)

    # PyTorch Lightning trainer with the best hyperparameters
    trainer = pl.Trainer(
        logger=TensorBoardLogger('tb_logs', name='TableNetExperiment'),
        max_epochs=10,  # Can be customized
        gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        callbacks=[
            ModelCheckpoint(monitor='validation_loss', dirpath=args.weights_dir),
            EarlyStopping(monitor='validation_loss', patience=10),
            LearningRateMonitor(logging_interval='step')
        ]
    )
    # Train the final model

    trainer.fit(model, datamodule=complaint_dataset)
    trainer.test()
