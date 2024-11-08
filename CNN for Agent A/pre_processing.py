import os
import numpy as np
import pickle
import shutil
from sklearn.model_selection import train_test_split

# Define paths to folders
data_folders = {
    'pdf': 'F:/Research work/Paid work-1/Implementation/Agent A/pdf',
    'images': 'F:/Research work/Paid work-1/Implementation/Agent A/images',
    'executables': 'F:/Research work/Paid work-1/Implementation/Agent A/executables'
}

# Label mapping
label_mapping = {'pdf': 0, 'images': 1, 'executables': 2}

# Output directories
output_dir = 'output_data'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Create directories for train and test sets
def create_dirs(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for label in label_mapping.keys():
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

create_dirs(train_dir)
create_dirs(test_dir)

# Create a list to store labeled data
labeled_data = []

# Function to read a file in binary mode
def read_file_as_binary(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

# Iterate through each folder and process files
for folder, path in data_folders.items():
    label = label_mapping[folder]
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            binary_data = read_file_as_binary(file_path)
            labeled_data.append((binary_data, label, filename))

# Split data into training and testing sets
train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42)

# Function to save binary data to respective folders
def save_binary_data(data, base_dir):
    for binary_data, label, filename in data:
        label_name = list(label_mapping.keys())[list(label_mapping.values()).index(label)]
        file_path = os.path.join(base_dir, label_name, filename + '.bin')
        with open(file_path, 'wb') as f:
            f.write(binary_data)

# Save training and testing data
save_binary_data(train_data, train_dir)
save_binary_data(test_data, test_dir)

print("Binary data conversion and separation into train/test completed.")
