from collections import OrderedDict
from typing import List
import os
import click
import cv2
import numpy as np
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from tablenet import TableNetModule

"""This script performs table and column detection on the given set of images. The script uses a pre-trained TableNet model to predict the locations of tables and columns 
in each image. It then draws bounding boxes around the predicted table and column areas and saves the annotated 
images in a specified output folder.

The script is designed to process multiple images located in a given folder. It uses albumentations for image 
transformations and OpenCV for drawing bounding boxes.
"""
def draw_lines(image, mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image, [contour], 0, color, 2)


class Predict:
    def __init__(self, checkpoint_path: str, transforms: Compose, threshold: float = 0.5, per: float = 0.005):
        self.transforms = transforms
        self.threshold = threshold
        self.per = per

        self.model = TableNetModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.requires_grad_(False)

    def predict_and_draw(self, image_path: str, output_path: str):
        image = cv2.imread(image_path)
        transformed = self.transforms(image=np.array(image))["image"]

        with torch.no_grad():
            pred_table, pred_column = self.model(transformed.unsqueeze(0))

        pred_table = (pred_table.squeeze().cpu().numpy() > self.threshold).astype(np.uint8)
        pred_column = (pred_column.squeeze().cpu().numpy() > self.threshold).astype(np.uint8)

        draw_lines(image, pred_table * 255, (0, 255, 0))  # Green for table
        draw_lines(image, pred_column * 255, (0, 0, 255))  # Red for column

        cv2.imwrite(output_path, image)


@click.command()
@click.option('--image_folder', default="./data/Marmot_data/")
@click.option('--model_weights', default="./data/best_model.ckpt")
@click.option('--output_folder', default="./data/Annotated_Marmot_data/")
def predict(image_folder: str, model_weights: str, output_folder: str):
    transforms = album.Compose([
        album.Resize(896, 896, always_apply=True),
        album.Normalize(),
        ToTensorV2()
    ])

    predictor = Predict(model_weights, transforms)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, filename)
            predictor.predict_and_draw(image_path, output_path)


if __name__ == '__main__':
    predict()
