import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import joblib

# Load the dataset
df = pd.read_csv('merged_cleaned.csv')
x_audio_processed = np.array([np.array(ast.literal_eval(features)) for features in df['audio-processed']])
x_additional_features = df[['mean_pitch', 'mean_energy', 'speech_rate']].values
x = np.hstack((x_audio_processed, x_additional_features))
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'ps': 5, 'sad': 6}
y = df['label_y'].replace(label_map).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")



# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)



# Save the trained model
joblib.dump(rf, 'emotion_detection_model.pkl')

