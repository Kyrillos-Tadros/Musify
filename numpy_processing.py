import librosa
import numpy as np

def load_and_process_audio(file_path, max_bins=128):
    """
    Load an audio file and extract mel-spectrogram, chroma, and MFCC features for music genre classification.
    
    Parameters:
    file_path (str): The path to the audio file.
    max_bins (int): The maximum number of frequency bins to consider.
    
    Returns:
    numpy.ndarray: The concatenated features (mel_db, chroma, mfcc).
    """
    try:
        # Load audio file with a consistent sampling rate
        y, sr = librosa.load(file_path, sr=44100)
        
        # Define parameters for consistency
        n_fft = 2048 # FFT window size
        hop_length = 512 # number of samples between successive frames
        
        # Extract Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=max_bins)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Extract Chroma Feature
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        
        # Standardize features
        mel_db_std = standardize_feature(mel_db)
        chroma_std = standardize_feature(chroma)
        mfcc_std = standardize_feature(mfcc)
        
        # Concatenate features along axis=0 (vertically)
        features = np.concatenate((mel_db_std, chroma_std, mfcc_std), axis=0)
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])

def standardize_feature(feature):
    """
    Standardize a feature by subtracting the mean and dividing by the standard deviation.
    
    Parameters:
    feature (numpy.ndarray): The feature to be standardized.
    
    Returns:
    numpy.ndarray: The standardized feature.
    """
    feature_mean = np.mean(feature, axis=1, keepdims=True)
    feature_std = np.std(feature, axis=1, keepdims=True)
    feature_std[feature_std == 0] = 1  # Avoid division by zero
    feature_std = (feature - feature_mean) / feature_std
    return feature_std