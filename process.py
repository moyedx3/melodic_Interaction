import os
import librosa
import numpy as np
import pandas as pd


# Set the path to the directory containing the audio files
audio_dir = './dataverse_files'

def load_audio_files(directory):
    audio_files = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            y, sr = librosa.load(filepath, sr=None)
            emotion = filename.split('_')[2]  # Assuming the emotion is the third part of the filename
            audio_files.append((y, sr))
            labels.append(emotion)
    return audio_files, labels

# Load the audio files and their labels
audio_files, labels = load_audio_files(audio_dir)

""" 
Display the first few loaded files and their labels
for i in range(3):
    print(f"Audio file {i+1}:")
    print(f"  Signal: {audio_files[i][0]}")
    print(f"  Sample rate: {audio_files[i][1]}")
    print(f"  Emotion: {labels[i]}")
"""
# Extract MFCCs
def extract_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).flatten()  # Flatten to keep it simple
    return mfccs

# Extract Pitch
def extract_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = [pitches[:, i].max() for i in range(pitches.shape[1])]
    return np.array(pitch).flatten()

# Extract Energy/Intensity
def extract_energy(y):
    energy = np.sum(librosa.feature.rms(y=y)**2)
    return energy

# Extract Speech-rate
def extract_speech_rate(y, sr):
    # This function estimates the speech rate as the number of beats per second
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    speech_rate = tempo / 60.0  # Convert to beats per second
    return speech_rate

# Initialize lists to store features
mfcc_features = []
pitch_features = []
energy_features = []
speech_rate_features = []

# Extract and store features for each audio file
for y, sr in audio_files:
    mfcc_features.append(extract_mfcc(y, sr))
    pitch_features.append(extract_pitch(y, sr))
    energy_features.append(extract_energy(y))
    speech_rate_features.append(extract_speech_rate(y, sr))

# Create a DataFrame
features_df = pd.DataFrame({
    'MFCCs': mfcc_features,
    'Pitch': pitch_features,
    'Energy': energy_features,
    'Speech_rate': speech_rate_features,
    'Emotion': labels
})

# Display the first few rows
print(features_df.head())