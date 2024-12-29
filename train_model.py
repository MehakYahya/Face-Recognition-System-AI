import os
import tensorflow as tf
from preprocess_data import preprocess_data

# Constants
IMG_SIZE = 128
MODEL_PATH = "../results/face_recognition_model.h5"

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=None):
    model_ = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),  # Softmax for multi-class
    ])
    model_.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model_

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    # Preprocess the data
    data_splits, label_map = preprocess_data()
    train_imgs = data_splits["train"]["images"]
    train_labels = data_splits["train"]["labels"]
    valid_imgs = data_splits["valid"]["images"]
    valid_labels = data_splits["valid"]["labels"]

    # Check that images and labels are non-empty before training the model
    if not all([len(train_imgs) > 0, len(train_labels) > 0, len(valid_imgs) > 0, len(valid_labels) > 0]):
        raise ValueError("One or more datasets are empty. Please check the preprocessing step.")

    # Number of classes
    num_classes = len(label_map)

    # Check if num_classes is valid
    if num_classes <= 1:
        raise ValueError("There should be at least two classes for classification.")

    # Build and train the model
    model = build_model(num_classes=num_classes)
    model.summary()

    # Use a smaller batch size and more epochs if needed
    model.fit(train_imgs, train_labels, batch_size=32, epochs=15, validation_data=(valid_imgs, valid_labels))

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
