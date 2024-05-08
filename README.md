**Overview**
This project is an adaptation of [OCR_tablenet by Tomás Sosorio](https://github.com/tomassosorio/OCR_tablenet) and utilizes the TableNet deep learning architecture for detecting tables and columns within document images. The implementation uses PyTorch Lightning for handling the training and evaluation process. The codebase is designed to process images and their corresponding labels, train the model, and make predictions on new images.

**Disclaimer**
This project is currently in the phase of adaptation and restructuring. While it is heavily based on OCR_tablenet by Tomás Sosorio, the names and overall code structure are in the process of being modified and adapted. The transition is being done in phases.


**Marmot.py:** This script handles the data loading and preprocessing. It reads images and their corresponding XML labels to create table and column masks.

**Tablenet.py:** This script contains the TableNet model architecture and the training, validation, and test steps.

**train.py:** This script is the main driver for training the model. It sets up the training environment, including data loading, model initialization, and training settings.

**predict.py:** This script is used for making predictions on new images. It applies the trained model to detect tables and columns in the images.

**visualize.py:** This script uses the trained model to make predictions and then draws bounding boxes around the predicted table and column areas. The annotated images are saved in a specified output folder.

**Prerequisites**
Python 3.x
PyTorch
PyTorch Lightning
albumentations
OpenCV
PIL
xml.etree.ElementTree
pandas


**How to Run**
Training
To train the model, navigate to the project directory and run:

python train.py --data_dir="./path/to/data" --weights_dir="./path/to/save/weights"

Prediction
To make predictions on new images, run:

python predict.py --image_path="./path/to/image" --model_weights="./path/to/weights"

Visualization
To visualize the model's predictions by drawing bounding boxes on the images, run:
python visualize.py --image_folder="./path/to/image/folder" --model_weights="./path/to/weights" --output_folder="./path/to/output/folder"

**Evaluation Metrics**
Dice Loss
Intersection over Union (IoU)




