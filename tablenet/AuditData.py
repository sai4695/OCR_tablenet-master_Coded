from pathlib import Path
from typing import List
import numpy as np
import pytorch_lightning as pl
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import xml.etree.ElementTree as ET


def create_mask_from_xml(xml_path, image_shape):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        table_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        column_mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for obj in root.iter('object'):
            name = obj.find('name').text
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)

            if name == 'table':
                cv2.rectangle(table_mask, (xmin, ymin), (xmax, ymax), 1, -1)
            elif name == 'column':
                cv2.rectangle(column_mask, (xmin, ymin), (xmax, ymax), 1, -1)

        return table_mask, column_mask
    except ET.ParseError:
        raise ValueError(f"Could not parse XML: {xml_path}")



class AuditImageDataset(Dataset):
    def __init__(self, data: List[Path], transforms: Compose = None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample_id = self.data[item].stem
        image_path = self.data[item].parent.parent.joinpath("labeled_images", sample_id + ".png")
        xml_path = self.data[item].parent.parent.joinpath("labels", sample_id + ".xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file does not exist: {xml_path}")
        image = np.array(Image.open(image_path))

        # Create masks using the function
        table_mask, column_mask = create_mask_from_xml(xml_path, image.shape)

        table_mask = np.expand_dims(table_mask, axis=2)
        column_mask = np.expand_dims(column_mask, axis=2)
        mask = np.concatenate([table_mask, column_mask], axis=2)

        sample = {"image": image, "mask": mask}
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)

        image = sample["image"]
        mask_table = sample["mask"][:, :, 0].unsqueeze(0)
        mask_column = sample["mask"][:, :, 1].unsqueeze(0)
        return image, mask_table, mask_column


class AuditImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", transforms_preprocessing: Compose = None,
                 transforms_augmentation: Compose = None, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.data = list(Path(data_dir).rglob("*.png"))
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage: str = None):
        n_samples = len(self.data)
        self.data.sort()
        train_slice = slice(0, int(n_samples * 0.8))
        val_slice = slice(int(n_samples * 0.8), int(n_samples * 0.9))
        test_slice = slice(int(n_samples * 0.9), n_samples)

        self.complaint_train = AuditImageDataset(self.data[train_slice], transforms=self.transforms_augmentation)
        self.complaint_val = AuditImageDataset(self.data[val_slice], transforms=self.transforms_preprocessing)
        self.complaint_test = AuditImageDataset(self.data[test_slice], transforms=self.transforms_preprocessing)

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.complaint_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.complaint_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.complaint_test, batch_size=self.batch_size, num_workers=self.num_workers)
