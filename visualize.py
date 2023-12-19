import os
import logging
import click
import cv2
import numpy as np
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from tablenet import TableNetModule
import torch
from albumentations import Compose

""" This script performs table and column detection on the given set of images. 
     The script uses a pre-trained TableNet model to predict the locations of tables
     and columns in each image. It then draws bounding boxes around the predicted
     table and column areas and saves the annotated 
images in a specified output folder.

The script is designed to process multiple images located in a given folder. It uses albumentations for image 
transformations and OpenCV for drawing bounding boxes.
"""


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def draw_lines(image, mask, color):
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for contour in contours:
            cv2.drawContours(image, [contour], 0, color, 2)
    except Exception as e:
        logger.error(f"Error drawing contours: {e}")
        raise


class Predict:
    def __init__(self, checkpoint_path: str, transforms: Compose, threshold: float = 0.5, per: float = 0.005):
        self.transforms = transforms
        self.threshold = threshold
        self.per = per

        try:
            self.model = TableNetModule.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.model.requires_grad_(False)
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            raise

    def predict_and_draw(self, image_path: str, output_path: str):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image not found or unable to read: {image_path}")
            transformed = self.transforms(image=np.array(image))["image"]
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

        try:
            with torch.no_grad():
                pred_table, pred_column = self.model(transformed.unsqueeze(0))
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            raise

        try:
            pred_table = (pred_table.squeeze().cpu().numpy() > self.threshold).astype(np.uint8)
            pred_column = (pred_column.squeeze().cpu().numpy() > self.threshold).astype(np.uint8)

            draw_lines(image, pred_table * 255, (0, 255, 0))  # Green for table
            draw_lines(image, pred_column * 255, (0, 0, 255))  # Red for column

            cv2.imwrite(output_path, image)
            logger.info(f"Annotated image saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error during drawing or saving annotated image: {e}")
            raise


@click.command()
@click.option('--image_folder', default="./data/Audit_data/")
@click.option('--model_weights', default="./data/best_model.ckpt")
@click.option('--output_folder', default="./data/Annotated_Audit_data/")
def predict(image_folder: str, model_weights: str, output_folder: str):
    transforms = album.Compose([
        album.Resize(896, 896, always_apply=True),
        album.Normalize(),
        ToTensorV2()
    ])

    try:
        predictor = Predict(model_weights, transforms)
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        return

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output folder: {e}")
        return

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                predictor.predict_and_draw(image_path, output_path)
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")


if __name__ == '__main__':
    predict()
