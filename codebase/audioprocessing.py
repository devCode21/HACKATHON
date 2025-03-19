import librosa
import os
import numpy as np
import pandas as pd

df=pd.read_csv('data1.csv')
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=None)  # Use the original sample rate of the file
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)  # Average over time

dataset = []


for i in df['speech']:
    dataset.append(extract_features(i))


dataset=np.asarray(dataset)

df1 =pd.DataFrame(df['speech'])
df1['']

print(dataset)











