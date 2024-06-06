import numpy as np
from dividing_data import divide_data
from three_seconds_segmentation import segment_music_files
from feature_extraction import extract_features_for_all_sets
from data_augmentation import augment_data
from model import initialize_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def load_audio_features():
    X_train = np.load('3_sec_features/Train_features.npy')
    y_train = np.load('3_sec_features/Train_labels.npy')
    X_test = np.load('3_sec_features/Test_features.npy')
    y_test = np.load('3_sec_features/Test_labels.npy')
    X_val = np.load('3_sec_features/Validation_features.npy')
    y_val = np.load('3_sec_features/Validation_labels.npy')
    return X_train, y_train, X_test, y_test, X_val, y_val



def train_model(X_train, y_train, X_val, y_val):
    # using data augmentation
    X_augmented, y_augmented = augment_data(X_train, y_train)

    # Get the input shape for model
    input_shape = X_augmented.shape[1:]
    input_shape = (input_shape[0], input_shape[1], 1)

    # Create the model
    model = initialize_model()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define the callbacks
    model_checkpoint = callbacks.ModelCheckpoint("model_best.keras", monitor='val_accuracy', save_best_only=True)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=0.00001)
    early_stopper = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_augmented, y_augmented, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32,
                    callbacks=[model_checkpoint, lr_reducer, early_stopper])
    
    # Save the final model
    model.save('musify_app.keras')
    
    return history



def main():
    # deviding data to train val and test 
    parent_folder = "genres_original"
    output_folder_name = "divided_files"
    divide_data(parent_folder, output_folder_name)

    # segmenting data into 3 seconds
    segment_music_files('divided_files', 'segmented_3')

    # extracting chroma, mel spectrogram and mfcc features, concatinate it together and save them 
    extract_features_for_all_sets(
        parent_dir="segmented_3",
        output_dir="3_sec_features"
    )

    # load audio features 
    X_train, y_train, X_test, y_test, X_val, y_val = load_audio_features()

    # Train the model
    history = train_model(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
