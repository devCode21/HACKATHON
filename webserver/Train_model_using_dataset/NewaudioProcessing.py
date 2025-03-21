import librosa
import pandas as pd
import os
import numpy as np
import kagglehub


path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

paths = []
labels = []
pitch_means = []
energy_means = []
speech_rates = []

# Process each audio file
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".wav"):  # Process only .wav files
            # Store the file path
            file_path = os.path.join(dirname, filename)
            paths.append(filename)
            
            # Extract the label from the filename (assuming it's the third part)
            label = filename.split('_')[2].split('.')[0]
            labels.append(label.lower())
            
            # Load the audio file
            y, sr = librosa.load(file_path, sr=16000)
            
            # **Pitch (Fundamental Frequency) Extraction**
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
            mean_pitch = np.nanmean(f0)  # Mean pitch (ignoring NaNs)
            pitch_means.append(mean_pitch)
            
            # **Energy Extraction (RMS Energy)**
            energy = librosa.feature.rms(y=y)
            mean_energy = np.mean(energy)  # Mean energy (RMS)
            energy_means.append(mean_energy)
            
            # **Speech Rate Extraction**
            # Estimate speech rate by counting frames with energy above a threshold
            speech_rate = np.sum(energy > 0.01)  # Adjust threshold as needed
            speech_rates.append(speech_rate)

            print(filename)

# Create a DataFrame to store the features
features_df = pd.DataFrame({
    'file_path': paths,
    'label': labels,
    'mean_pitch': pitch_means,
    'mean_energy': energy_means,
    'speech_rate': speech_rates
})

# Print the DataFrame
print(features_df.head())  # Print the first few rows of the DataFrame to check the results

features_df.to_csv('NewData.csv')


