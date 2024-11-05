import os
import numpy as np
import cv2  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os

train_data_dir = "C:/Users/Mohin/OneDrive/Desktop/VSCode/project/data/train"

# Print the contents of the train directory
print("Contents of train directory:")
for label in os.listdir(train_data_dir):
    label_dir = os.path.join(train_data_dir, label)
    if os.path.isdir(label_dir):
        num_images = len(os.listdir(label_dir))
        print(f"Class: {label}, Number of images: {num_images}")
    else:
        print(f"{label} is not a directory.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def load_and_preprocess_data(data_dir):
    X, y = [], []
    label_map = {}
    current_label = 0

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            print(f"Processing label: {label}")
            if label not in label_map:
                label_map[label] = current_label
                current_label += 1
            for filename in os.listdir(label_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(label_dir, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Warning: Unable to load image {image_path}. Skipping.")
                        continue
                    image = cv2.resize(image, (64, 64))
                    X.append(image)
                    y.append(label_map[label])  # Map the label name to an integer
                    print(f"Loaded image: {image_path}")
                    
    if not X:  # Check if X is empty
        print("No images found in the dataset.")
    
    X = np.array(X) / 255.0  # Normalize
    y = to_categorical(y, num_classes=current_label)  # One-hot encode labels
    return X[..., np.newaxis], y  # Add channel dimension

def create_cnn_model(input_shape=(64, 64, 1), num_classes=10):
    """Create and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_data_dir, model_path):
    """Load training data, train the model, and save it."""
    X_train, y_train = load_and_preprocess_data(train_data_dir)
    
    # Check if there are any images loaded
    if X_train.size == 0 or y_train.size == 0:
        print("No training data found. Please check your data directory.")
        return
    
    num_classes = len(os.listdir(train_data_dir))
    model = create_cnn_model(num_classes=num_classes)
    
    # Fit the model with training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    model.save(model_path)
    print("Model training complete and saved.")

if __name__ == "__main__":

    # Use absolute paths
   model_path = os.path.abspath("C:/Users/Mohin/OneDrive/Desktop/VSCode/project/model/trained_model.h5")
train_model(train_data_dir, model_path)