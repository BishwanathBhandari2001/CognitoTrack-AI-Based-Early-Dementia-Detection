# import sounddevice as sd
# from scipy.io.wavfile import write
# import numpy as np
# import librosa
# import os
# from tensorflow.keras.models import load_model

# # -------- settings --------
# DURATION = 10  # seconds
# FS = 22050
# TEMP_FILE = "live_input.wav"

# model = load_model("../models/bilstm_speakerwise.keras")


# def extract_logmel_sequence(path, max_len=300):
#     # EXACT SAME AS predict_audio.py
#     y, sr = librosa.load(path, duration=30)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
#     log_mel = librosa.power_to_db(mel).T

#     if len(log_mel) < max_len:
#         pad = np.zeros((max_len - len(log_mel), 64))
#         log_mel = np.vstack((log_mel, pad))
#     else:
#         log_mel = log_mel[:max_len]

#     return log_mel


# print("🎤 Recording for 10 seconds... Speak now!")
# recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
# sd.wait()

# # Proper WAV save
# write(TEMP_FILE, FS, (recording * 32767).astype(np.int16))

# print("🧠 Analyzing speech...")

# features = extract_logmel_sequence(TEMP_FILE)
# features = np.expand_dims(features, axis=0)

# prediction = model.predict(features)[0]
# control_prob = prediction[0]
# dementia_prob = prediction[1]

# print("\n🧠 Live Speech Analysis")
# print("━━━━━━━━━━━━━━━━━━━━━━")
# print(f"📊 Control Probability : {control_prob:.4f}")
# print(f"📊 Dementia Probability: {dementia_prob:.4f}")
# print("━━━━━━━━━━━━━━━━━━━━━━")

# # ✅ SAME RULE AS predict_audio.py (threshold kept)
# if dementia_prob > 0.10 and dementia_prob > control_prob:
#     print("⚠️ Result: DEMENTIA pattern detected")
# else:
#     print("✅ Result: CONTROL pattern detected")

# print("━━━━━━━━━━━━━━━━━━━━━━\n")

# # Cleanup
# if os.path.exists(TEMP_FILE):
#     os.remove(TEMP_FILE)

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# -------- SETTINGS --------
DURATION = 25        # IMPORTANT: closer to training audio length
FS = 22050
TEMP_FILE = "live_input.wav"

model = load_model("../models/bilstm_speakerwise.keras")


# -------- SAME FEATURE EXTRACTION AS TRAINING --------
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


# -------- RECORD LIVE AUDIO --------
print("🎤 Recording for 25 seconds... Speak slowly with pauses!")
print("Example: describe a scene with hesitation and pauses.\n")

recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
sd.wait()

# ✅ Normalize amplitude like dataset audio
recording = recording / np.max(np.abs(recording))

# Save proper WAV
write(TEMP_FILE, FS, (recording * 32767).astype(np.int16))

print("🧠 Analyzing speech...\n")

# -------- PREDICTION --------
features = extract_logmel_sequence(TEMP_FILE)
features = np.expand_dims(features, axis=0)

prediction = model.predict(features)[0]
control_prob = float(prediction[0])
dementia_prob = float(prediction[1])

print("🧠 Live Speech Analysis")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"📊 Control Probability : {control_prob:.4f}")
print(f"📊 Dementia Probability: {dementia_prob:.4f}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ✅ EXACT SAME RULE AS predict_audio.py
if control_prob > dementia_prob:
    print("✅ Result: CONTROL (Healthy Speech Pattern)")
else:
    print("⚠️ Result: DEMENTIA (Hesitant / Paused Speech Pattern)")

print("━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# -------- CLEANUP --------
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)
