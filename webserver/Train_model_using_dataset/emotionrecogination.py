import kagglehub
import os
import pandas as pd


path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")


paths = []
labels = []

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".wav"): 
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[2].split('.')[0]
            labels.append(label.lower()) 

print(f'Dataset Loaded with {len(paths)} audio files')

df = pd.DataFrame({'speech': paths, 'label': labels})


print(df.head())

# Save the DataFrame to CSV
df.to_csv('data1.csv', index=False)
