import os
import soundfile as sf
import librosa

def segment_music_files(input_path, output_dir, segment_duration=3):
    """
    Segments the audio files in the input path and copies the segmented files
    to the output directory, maintaining the same folder structure.

    Args:
        input_path (str): The path to the input file or directory containing the audio files.
        output_dir (str): The path to the output directory where the segmented files will be copied.
        segment_duration (int, optional): The duration of each audio segment in seconds. Defaults to 3.
    """
    # Check if the input path is a file or a directory
    if os.path.isfile(input_path):
        # Handle a single file
        segment_single_file(input_path, output_dir, segment_duration)
    else:
        # Handle a directory structure
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                if filename.lower().endswith('.wav'):
                    file_path = os.path.join(root, filename)
                    try:
                        # Load the audio file using librosa
                        audio_file, sr = librosa.load(file_path)

                        # Create the output subfolder if it doesn't exist
                        relative_path = os.path.relpath(root, input_path)
                        output_subfolder = os.path.join(output_dir, relative_path)
                        os.makedirs(output_subfolder, exist_ok=True)

                        # Loop through the audio data in 3-second chunks
                        segment_count = 0
                        for i in range(0, len(audio_file), int(segment_duration * sr)):
                            segment = audio_file[i:i + int(segment_duration * sr)]

                            # Check if the segment is non-empty before saving
                            if len(segment) >= int(segment_duration * sr) and len(segment) > 0:
                                # Create a new filename for the segment
                                new_filename = f"{os.path.splitext(filename)[0]}_segment_{segment_count}.wav"
                                new_file_path = os.path.join(output_subfolder, new_filename)
                                # Save the segment to the output folder
                                sf.write(new_file_path, segment, sr)
                                segment_count += 1
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")

def segment_single_file(input_file, output_dir, segment_duration=3):
    """
    Segments a single audio file into 3-second chunks and saves them in the output directory.

    Args:
        input_file (str): The path to the input audio file.
        output_dir (str): The path to the output directory where the segmented files will be saved.
        segment_duration (int, optional): The duration of each audio segment in seconds. Defaults to 3.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio file
    audio_file, sr = librosa.load(input_file)

    # Loop through the audio data in 3-second chunks
    segment_count = 0
    for i in range(0, len(audio_file), int(segment_duration * sr)):
        segment = audio_file[i:i + int(segment_duration * sr)]

        # Check if the segment is non-empty before saving
        if len(segment) >= int(segment_duration * sr) and len(segment) > 0:
            # Create a new filename for the segment
            new_filename = f"segment_{segment_count}.wav"
            new_file_path = os.path.join(output_dir, new_filename)
            # Save the segment to the output folder
            sf.write(new_file_path, segment, sr)
            segment_count += 1
