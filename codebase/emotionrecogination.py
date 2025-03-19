import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings

warnings.filterwarnings('ignore')

import kagglehub

labels = []
paths = []

# Download latest version of the TESS dataset
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

# Assuming the dataset is extracted in the directory specified by kagglehub
# Traverse the dataset directory
for dirname, _, filenames in os.walk(path):  # Using path where the dataset is downloaded
    for filename in filenames:
        if filename.endswith(".wav"):  # Make sure to process only .wav files
            paths.append(os.path.join(dirname, filename))
            # data\OAF_angry\OAF_back_angry.wav
            # Extract emotion label from the filename
            # Assuming the format: 'subject_emotion_XX.wav' (you may adjust the split logic)
            label = filename.split('_')[2].split('.')[0]
            print(f'filename is {filename},label is {label}')
        
            labels.append(label.lower())  # Append the label in lowercase
            
    # You may want to break after collecting all the relevant files or process all files in the dataset
    # No need for 'len(paths) == 2800' unless you know the exact number of files
    # if len(paths) == 2800:
    #     break

print(f'Dataset Loaded with {len(paths)} audio files')

print(len(paths))
print(labels[:5])


df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

print(df['label'].value_counts())

df.to_csv('data1.csv')
