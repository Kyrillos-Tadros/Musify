import os
import numpy as np
from numpy_processing import load_and_process_audio, standardize_feature
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tqdm import tqdm

def extract_audio_features(data_dir, output_dir, max_bins=128):
    """
    Extracts robust audio features from audio files in a directory with nested subfolders.
    
    Parameters:
    data_dir (str): The path to the directory containing the Train, Test, and Validation subdirectories.
    output_dir (str): The path to the directory where the features and labels will be saved.
    max_bins (int): The maximum number of frequency bins to consider.
    
    Returns:
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()
    
    # Iterate over Train, Test, and Validation subdirectories
    for subset in ['Train', 'Test', 'Validation']:
        subset_dir = os.path.join(data_dir, subset)
        subset_data = []
        subset_labels = []
        
        # Iterate over all genre subfolders
        for genre_dir in os.listdir(subset_dir):
            genre_label = genre_dir
            subdirectory_path = os.path.join(subset_dir, genre_dir)

            for filename in tqdm(os.listdir(subdirectory_path), desc=f"Processing files in {genre_dir}"):
                filepath = os.path.join(subdirectory_path, filename)
                try:
                    features = load_and_process_audio(filepath, max_bins)
                    
                    if features.size > 0:
                        subset_data.append(features)
                        subset_labels.append(genre_label)
                except UserWarning as e:
                    if "Trying to estimate tuning from empty frequency set" in str(e):
                        print(f"Error processing file {filename} in {genre_dir}: {e}")
                        print(filepath)

        # Convert data and labels to numpy arrays
        subset_data_np = np.array(subset_data)
        subset_labels_np = np.array(subset_labels)
        
        # Encode string labels to numerical values
        subset_labels_encoded = label_encoder.fit_transform(subset_labels_np)
        
        # Convert labels to categorical format
        num_classes = len(label_encoder.classes_)
        subset_labels_categorical = to_categorical(subset_labels_encoded, num_classes)
        
        # Save features and labels for the current subset
        save_path_features = os.path.join(output_dir, f"{subset}_features.npy")
        save_path_labels = os.path.join(output_dir, f"{subset}_labels.npy")
        np.save(save_path_features, subset_data_np)
        np.save(save_path_labels, subset_labels_categorical)