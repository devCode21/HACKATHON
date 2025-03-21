import kagglehub
import os
import pandas as pd

# Download the dataset (you already have this step)
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")


paths = []
labels = []

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".wav"):  # Process only .wav files
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[2].split('.')[0]
            labels.append(label.lower())  # Label is the third part of the filename

print(f'Dataset Loaded with {len(paths)} audio files')

# Create a DataFrame with paths and labels
df = pd.DataFrame({'speech': paths, 'label': labels})

# Print the first few rows
print(df.head())

# Save the DataFrame to CSV
df.to_csv('data1.csv', index=False)
