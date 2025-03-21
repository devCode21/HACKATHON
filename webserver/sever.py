import os
import librosa
import numpy as np
import joblib
import uuid
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained model (ensure this path is correct)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'codebase', 'emotion_detection_model.pkl')
model = joblib.load(model_path)

def extract_features(audio_file):
    try:
        print(f"Loading audio file: {audio_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file {audio_file} does not exist.")

        # Step 1: Load the audio file
        y, sr = librosa.load(audio_file, sr=16000)
        print(f"Audio loaded. Sample rate: {sr}, Audio shape: {y.shape}")

        # Step 2: Trim the audio (remove silence at the beginning and end)
        y, _ = librosa.effects.trim(y)
        print(f"Audio trimmed. New shape: {y.shape}")

        # Step 3: Normalize the audio
        y = librosa.util.normalize(y)

        # Step 4: Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        print(f"MFCC extracted. Shape: {mfcc_mean.shape}")

        # Step 5: Extract pitch (f0)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        mean_pitch = np.mean(f0) if f0 is not None else 0
        print(f"Pitch extracted. Mean pitch: {mean_pitch}")

        # Step 6: Extract RMS energy
        energy = librosa.feature.rms(y=y)
        mean_energy = np.mean(energy)
        print(f"RMS energy extracted. Mean energy: {mean_energy}")

        # Step 7: Estimate speech rate
        speech_rate = np.sum(energy > 0.01)
        print(f"Speech rate estimated. Value: {speech_rate}")

        # Step 8: Combine all features into a single array
        features = np.hstack([mfcc_mean, mean_pitch, mean_energy, speech_rate])
        print(f"Features combined. Total feature length: {features.shape}")

        return features

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise ValueError(f"Error during feature extraction: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file found in the request'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Only .wav files are allowed'}), 400

        temp_audio_path = f'temp_audio_{uuid.uuid4().hex}.wav'
        audio_file.save(temp_audio_path)

        if not os.path.exists(temp_audio_path):
            raise FileNotFoundError(f"Failed to save the file {temp_audio_path}.")

        features = extract_features(temp_audio_path)
        if features is None:
            return jsonify({'error': 'Failed to extract features from the audio'}), 400

        features = features.reshape(1, -1)
        prediction = model.predict(features)
        label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
        predicted_label = label_map[prediction[0]]

        os.remove(temp_audio_path)

        return jsonify({'predicted_label': predicted_label})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Error processing the request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
