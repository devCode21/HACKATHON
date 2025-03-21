import numpy as np
import pandas as pd
import ast
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# Load the pre-existing dataset
df = pd.read_csv('merged_cleaned.csv')

# Remove rows where 'audio-processed' is NaN
df = df.dropna(subset=['audio-processed'])

# Process the feature column and labels
x_audio_processed = np.array([np.array(ast.literal_eval(features)) for features in df['audio-processed']])
x_additional_features = df[['mean_pitch', 'mean_energy', 'speech_rate']].values

# Impute missing values in the additional features (numerical columns)
imputer = SimpleImputer(strategy='mean')
x_additional_features = imputer.fit_transform(x_additional_features)

# Combine the features
x = np.hstack((x_audio_processed, x_additional_features))

# Impute any remaining NaN values in the final feature set
x_imputed = SimpleImputer(strategy='mean').fit_transform(x)

# Process the labels
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'ps': 5, 'sad': 6}
y = df['label_y'].replace(label_map).values

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.2, random_state=42)

# Scale the features for SVM (SVM performs better with scaled data)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyperparameter tuning using GridSearchCV for better model performance
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 5]
}

# Perform GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Get the best model from GridSearchCV
best_svm_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_svm_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Emotion Detection Accuracy (after tuning): {accuracy * 100:.2f}%")

# Save the best model and the scaler for later use
joblib.dump(best_svm_model, 'emotion_svm_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for real-time scaling

# Optionally, you can also train a RandomForest as an alternative:
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Emotion Detection Accuracy: {accuracy_rf * 100:.2f}%")
