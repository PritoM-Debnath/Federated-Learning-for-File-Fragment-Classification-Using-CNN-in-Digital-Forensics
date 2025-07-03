# Federated Learning for File Fragment Classification Using CNN in Digital Forensics Data
## Overview
This project implements a federated learning approach for file fragment classification using Convolutional Neural Networks (CNN) in the field of digital forensics. It aims to classify file fragments into categories such as PDFs, images, and executables based on their binary data.

## Features
- **Data Preprocessing**: Converts raw files into binary data and organizes them into train/test sets.
- **Model Training**: Utilizes a 1D CNN architecture to train and classify file fragments.
- **Evaluation**: Evaluates model performance using accuracy metrics.
- **Visualization**: Provides visualizations of training and validation metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PritoM-Debnath/Federated-Learning-for-File-Fragment-Classification-Using-CNN-in-Digital-Forensics.git
   cd your_repository
    ```
2. Set up Python environment:
   ```bash
   pip install -r requirements.txt
    ```
3. Run preprocessing script to convert data:
   ```bash
   python pre_processing.py
   ```
4. Train and evaluate the model:
   ```bash
   python training.py
   ```
## Usage
- Modify paths and configurations in pre-processing.py and training.py as per your environment and data location.
- Adjust parameters in the CNN model defined in training.py for experimentation.

