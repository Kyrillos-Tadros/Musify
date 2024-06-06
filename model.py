from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, callbacks
import numpy as np



def initialize_model():
    model = Sequential()

    # Input layer
    model.add(layers.Input(shape=(153, 259, 1)))

    # Convolutional layers with Batch Normalization and increased filters
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Dropout(0.6))

    # Flatten the output of the conv layers to feed into the dense layers
    model.add(layers.Flatten())

    # Dense layers with Batch Normalization
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())


    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.6))

    # Output layer with the number of classes
    num_classes = 10  # Update with the number of music genres you have
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model



