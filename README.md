# Musify: Music Genre Classification

Musify is a project focused on music genre classification. It uses the GTZAN dataset, a popular dataset in the field of music information retrieval. The GTZAN dataset consists of 1,000 audio tracks, each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22,050Hz Mono 16-bit audio files in WAV format.

To avoid data leakage, the GTZAN dataset is manually divided into training, validation, and testing sets. This ensures that no song appears in both the training and testing sets after the audio files are segmented into three-second clips.

The methodology of the project is as follows:

1. **Data Preparation**: The GTZAN dataset is manually divided into training, validation, and testing sets. The audio files are then segmented into three-second clips to increase the volume of data.

2. **Feature Extraction**: Features such as Mel Spectrogram, Chroma, and MFCC (Mel-frequency cepstral coefficients) are extracted from the audio clips and saved as numpy arrays.

3. **Data Augmentation**: To enhance the robustness of the model, data augmentation techniques are applied to the extracted features.

4. **Model Training**: The augmented data is used to train the classification model.

5. **Deployment**: A Streamlit app is created for easy interaction with the model.

The project requires the following Python libraries:

- numpy
- soundfile
- librosa
- streamlit
- pytube
- tensorflow

After installing the required libraries, you can clone the repository and run the Streamlit app:

```
git clone https://github.com/Kyrillos-Tadros/musify.git
cd musify
streamlit run main.py
```

If you have suggestions for improving this project, please open an issue or submit a pull request.
