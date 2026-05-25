import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# ✅ correct model file
model = load_model("../models/bilstm_speakerwise.keras")

def extract_logmel_sequence(path, max_len=300):
    y, sr = librosa.load(path, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel).T

    if len(log_mel) < max_len:
        pad = np.zeros((max_len - len(log_mel), 64))
        log_mel = np.vstack((log_mel, pad))
    else:
        log_mel = log_mel[:max_len]

    return log_mel


audio_path = input("\n🎧 Enter WAV file path: ")

features = extract_logmel_sequence(audio_path)
features = np.expand_dims(features, axis=0)

prediction = model.predict(features)[0]
control_prob = prediction[0]
dementia_prob = prediction[1]

print("\n🧠 AI Speech Analysis Report")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"🔊 File: {os.path.basename(audio_path)}")
print(f"📊 Control Probability : {control_prob:.4f}")
print(f"📊 Dementia Probability: {dementia_prob:.4f}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ✅ correct decision rule
THRESHOLD = 0.5

if dementia_prob > THRESHOLD:
    print("🧠 Result: ⚠️  DEMENTIA DETECTED")
else:
    print("🧠 Result: ✅ CONTROL (Healthy Speech)")

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
