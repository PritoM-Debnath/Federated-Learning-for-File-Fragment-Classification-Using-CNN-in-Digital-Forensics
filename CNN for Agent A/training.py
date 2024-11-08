import os
import numpy as np
import keras
from keras import Sequential, layers
import matplotlib.pyplot as plt

# Define paths to train and test directories
train_dir = 'output_data/train'
test_dir = 'output_data/test'

# Label mapping
label_mapping = {'pdf': 0, 'images': 1, 'executables': 2}

# Maximum length for binary data
MAX_LENGTH = 250000  # Reduced to 500 KB to handle memory issues

# Function to load binary data and labels from a directory
def load_data_from_dir(base_dir):
    data = []
    labels = []
    for label_name, label_value in label_mapping.items():
        folder_path = os.path.join(base_dir, label_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as file:
                    binary_data = np.frombuffer(file.read(), dtype=np.uint8)
                    # Truncate if data is longer than MAX_LENGTH
                    if len(binary_data) > MAX_LENGTH:
                        binary_data = binary_data[:MAX_LENGTH]
                    data.append(binary_data)
                    labels.append(label_value)
    return data, labels

# Load training and testing data
X_train, y_train = load_data_from_dir(train_dir)
X_test, y_test = load_data_from_dir(test_dir)

# Pad data to MAX_LENGTH
X_train = np.array([np.pad(x, (0, max(0, MAX_LENGTH - len(x))), 'constant') for x in X_train])
X_test = np.array([np.pad(x, (0, max(0, MAX_LENGTH - len(x))), 'constant') for x in X_test])

# Reshape data to fit into the CNN input format
X_train = X_train.reshape((X_train.shape[0], MAX_LENGTH, 1))
X_test = X_test.reshape((X_test.shape[0], MAX_LENGTH, 1))
y_train = np.array(y_train)
y_test = np.array(y_test)

# Build a simpler 1D CNN model to reduce memory usage
model = Sequential([
    layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(MAX_LENGTH, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(label_mapping), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with a smaller batch size
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=8,  # Reduced batch size to manage memory usage
    validation_split=0.2
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
