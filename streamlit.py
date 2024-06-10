import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
import streamlit as st
from keras.models import load_model
from numpy_processing import load_and_process_audio
from three_seconds_segmentation import segment_music_files
from pytube import YouTube
from io import BytesIO
import dvc.api





# CSS styling
background_css = """
<style>
body {
  background-image: url('https://cdn.pixabay.com/photo/2016/01/14/06/09/woman-1139397_960_720.jpg');
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  opacity: 0.8; /* Adjust the opacity as needed */
  color: white; /* Set text color to white */
  font-size: 2em; /* Increase font size */
  padding: 20px; /* Add padding for better visibility */
}
p{
  color:black;
}
hr {
  margin: 0px 0px;   
  }
  h1{
    padding: 0px;
  }
 .stTabs{
      margin-top:-75px;
  }
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Main Page", "How to Use the App", "Genre Information"])

genre_info = {
    "blues": "Blues is a genre of music that originated in the African American communities of the United States. It typically features melancholy lyrics and a distinctive musical style.",
    "classical": "Classical music is a genre that encompasses a broad range of music from the Western tradition. It often features complex compositions and instrumental arrangements.",
    "country": "Country music is a genre that originated in the Southern United States. It often highlights themes of rural life, love, and heartache.",
    "disco": "Disco is a genre of dance music that was popular in the 1970s. It is characterized by a strong beat and electronic instrumentation.",
    "hip hop": "Hip hop is a genre of music that emerged in the Bronx, New York City, during the 1970s. It is characterized by rhythmic speech over a beat.",
    "jazz": "Jazz is a genre of  music that originated in the African American communities of New Orleans. It features improvisation and swing rhythms.",
    "metal": "Metal is a genre of music that is characterized by its loud, aggressive sound. It often features distorted guitars and powerful vocals.",
    "pop": "Pop music is a genre that encompasses popular music from various styles. It is often catchy and appeals to a broad audience.",
    "reggae": "Reggae is a genre of music that originated in Jamaica. It is characterized by its offbeat rhythms and socially conscious lyrics.",
    "rock": "Rock music is a genre that emerged in the 1950s and has since evolved into various subgenres. It typically features electric guitars and strong rhythms."
}

# Load the trained model
with dvc.api.open('my_model.h5', remote='myremote') as fd:
    model = load_model(fd)

# Define the genre labels
GENRES = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "Hiphop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

def download_audio_to_buffer(url):
    buffer = BytesIO()
    youtube_video = YouTube(url)
    audio = youtube_video.streams.get_audio_only()
    audio.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer

with tab1:
    st.markdown("<h1 style='text-align: center; font-size: 1.5em;color: black;margin-top:-15px;'>Musify</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;color: black;margin-top: -19px;font-size: 0.9em;'>Genre Classification App</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 1px solid #000;color: black;'>", unsafe_allow_html=True)

    # User choice
    choice = st.radio("Select an option:", ("Upload a file", "Enter a YouTube URL"))

    if choice == "Upload a file":
        # File upload
        uploaded_file = st.file_uploader("Upload a music file for genre classification:")

        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())

            # Create a temporary directory for segmentation
            temp_dir = "temp_segments"
            os.makedirs(temp_dir, exist_ok=True)

            # Segment the uploaded file into 3-second chunks
            segment_music_files(temp_file_path, temp_dir, segment_duration=3)

            # Extract features from the audio segments
            audio_features = []
            for filename in os.listdir(temp_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(temp_dir, filename)
                    features = load_and_process_audio(file_path)
                    if features.size > 0:
                        audio_features.append(features)

            audio_features_np = np.array(audio_features)

            # Make predictions
            predictions = model.predict(audio_features_np)

            # Calculate the cumulative probabilities for each genre
            genre_probabilities = np.sum(predictions, axis=0)

            # Determine the most likely genre
            most_likely_genre_index = np.argmax(genre_probabilities)
            most_likely_genre = GENRES[most_likely_genre_index]

            st.write(f"# Predicted Genre: {most_likely_genre}")
            st.markdown(genre_info[most_likely_genre.lower()])
            st.audio(uploaded_file)

            # Remove the temporary file and directory
            os.remove(temp_file_path)
            import shutil
            shutil.rmtree(temp_dir)

    elif choice == "Enter a YouTube URL":
        # YouTube URL input
        youtube_url = st.text_input("Enter a YouTube video URL:", placeholder="Add URL here")

        if youtube_url:
            # Download the audio from the YouTube video
            try:
                buffer = download_audio_to_buffer(youtube_url)

                # Create a temporary directory for segmentation
                temp_dir = "temp_segments"
                os.makedirs(temp_dir, exist_ok=True)

                # Save buffer to a temporary file
                temp_file_path = os.path.join(temp_dir, "temp_audio.mp3")
                with open(temp_file_path, "wb") as f:
                    f.write(buffer.getbuffer())

                # Segment the downloaded audio into 3-second chunks
                segment_music_files(temp_file_path, temp_dir, segment_duration=3)

                # Extract features from the audio segments
                audio_features = []
                for filename in os.listdir(temp_dir):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(temp_dir, filename)
                        features = load_and_process_audio(file_path)
                        if features.size > 0:
                            audio_features.append(features)

                audio_features_np = np.array(audio_features)

                # Make predictions
                predictions = model.predict(audio_features_np)

                # Calculate the cumulative probabilities for each genre
                genre_probabilities = np.sum(predictions, axis=0)

                # Determine the most likely genre
                most_likely_genre_index = np.argmax(genre_probabilities)
                most_likely_genre = GENRES[most_likely_genre_index]

                st.write(f"# Predicted Genre: {most_likely_genre}")
                st.markdown(genre_info[most_likely_genre.lower()])

                # Embed the YouTube video player using st.video
                st.video(youtube_url)

                # Remove the temporary file and directory
                os.remove(temp_file_path)
                import shutil
                shutil.rmtree(temp_dir)

            except Exception as e:
                st.error(f"Error processing YouTube URL: {e}")

with tab2:
    st.markdown("""
    <h2 style='color: black; font-size: 21px;'>How to Use the App</h2>
    <p style='color: black; font-size: 18px;'>1. Select your preferred option:</p>
    <ul style='color: black; font-size: 18px;'>
        <li><strong>Upload a file:</strong> Click the "Browse files" button and select the audio file you want to classify.</li>
        <li><strong>Enter a YouTube URL:</strong> Type the URL of the YouTube video in the provided text input field.</li>
    </ul>
    
    <p style='color: black; font-size: 18px;'>2. Wait for classification:</p>
    <p style='color: black; font-size: 18px;'>Once you provide the input (file or URL), the app will process it and predict the music genre. This might take a few seconds depending on the file size or network conditions.</p>
    
    <p style='color: black; font-size: 18px;'>3. View results:</p>
    <p style='color: black; font-size: 18px;'>After processing, the app will display the predicted genre, along with a brief description of the genre.</p>
    <ul style='color: black; font-size: 18px;'>
        <li>For an uploaded file, the audio player will be shown.</li>
        <li>For a YouTube URL, the embedded video player will be displayed.</li>
    </ul>
    """, unsafe_allow_html=True)

with tab3:
    genre_label = st.subheader("**Select a Genre to Learn More About:**")
    genre_dropdown = st.selectbox("", list(genre_info.keys()))

    if genre_dropdown:
        st.markdown(genre_info[genre_dropdown]) 
