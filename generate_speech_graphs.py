import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# =====================================================
# PATH SETTINGS
# =====================================================
DATA_PATH = "../data"
RESULTS_PATH = "../results"
MODEL_PATH = "../models/bilstm_speakerwise.keras"

os.makedirs(RESULTS_PATH, exist_ok=True)

print("\n📊 Generating Professional IEEE Graphs...\n")

# =====================================================
# LOAD TRAINED MODEL
# =====================================================
model = load_model(MODEL_PATH)

# =====================================================
# FEATURE EXTRACTION (SAME AS TRAINING)
# =====================================================
def extract_logmel(path, max_len=300):
    y, sr = librosa.load(path, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel).T

    if len(log_mel) < max_len:
        pad = np.zeros((max_len - len(log_mel), 64))
        log_mel = np.vstack((log_mel, pad))
    else:
        log_mel = log_mel[:max_len]

    return log_mel

# =====================================================
# FIND SAMPLE FILE
# =====================================================
def find_sample(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                return os.path.join(root, f)
    return None

# =====================================================
# FEATURE VISUALIZATION (LogMel + MFCC)
# =====================================================
def save_feature_images(wav_path, tag):
    y, sr = librosa.load(wav_path, duration=20)

    # ----- Log-Mel Spectrogram -----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel)

    plt.figure(figsize=(8,4))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-Mel Spectrogram ({tag})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency Bands")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/logmel_{tag}.png", dpi=300)
    plt.close()

    # ----- MFCC -----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    plt.figure(figsize=(8,4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f"MFCC Features ({tag})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/mfcc_{tag}.png", dpi=300)
    plt.close()

print("✔ Generating Feature Graphs...")

for label in ["control", "dementia"]:
    sample = find_sample(os.path.join(DATA_PATH, label))
    if sample:
        save_feature_images(sample, label)

# =====================================================
# PREPARE TEST DATA
# =====================================================
X_test = []
y_true = []

for label, cls in [("control",0), ("dementia",1)]:
    folder = os.path.join(DATA_PATH, label)
    count = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".wav") and count < 20:
                X_test.append(extract_logmel(os.path.join(root, f)))
                y_true.append(cls)
                count += 1

X_test = np.array(X_test)

# =====================================================
# PREDICTION
# =====================================================
pred_probs = model.predict(X_test)
pred = np.argmax(pred_probs, axis=1)
confidence = np.max(pred_probs, axis=1)

accuracy = np.mean(pred == y_true) * 100
print(f"✔ Test Accuracy used for graphs: {accuracy:.2f}%")

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["Control","Dementia"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - BiLSTM Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/confusion_matrix.png", dpi=300)
plt.close()

# =====================================================
# TEST CONFIDENCE CURVE
# =====================================================
plt.figure(figsize=(8,4))
plt.plot(confidence, marker='o')
plt.title("Test Prediction Confidence Curve")
plt.xlabel("Test Sample Index")
plt.ylabel("Confidence Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/test_graph.png", dpi=300)
plt.close()

# =====================================================
# PIE CHART
# =====================================================
correct = np.sum(pred == y_true)
incorrect = len(y_true) - correct

plt.figure()
plt.pie([correct, incorrect],
        labels=["Correct","Incorrect"],
        autopct="%1.1f%%",
        startangle=90)
plt.title("Prediction Distribution")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/pie_chart.png", dpi=300)
plt.close()

# =====================================================
# BAR GRAPH
# =====================================================
plt.figure()
plt.bar(["BiLSTM Test Accuracy"], [accuracy])
plt.ylim(0,100)
plt.ylabel("Accuracy (%)")
plt.title("Model Performance")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/bar_graph.png", dpi=300)
plt.close()

# =====================================================
# LINE CHART (Probability Trend)
# =====================================================
plt.figure(figsize=(8,4))
plt.plot(pred_probs[:,0], label="Control Probability")
plt.plot(pred_probs[:,1], label="Dementia Probability")
plt.legend()
plt.title("Prediction Probability Trends")
plt.xlabel("Test Sample Index")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/line_chart.png", dpi=300)
plt.close()

# =====================================================
# HISTOGRAM
# =====================================================
plt.figure()
plt.hist(confidence, bins=15)
plt.title("Confidence Score Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/histogram.png", dpi=300)
plt.close()

# =====================================================
# SCATTER PLOT
# =====================================================
plt.figure()
plt.scatter(range(len(confidence)), confidence)
plt.title("Confidence Scatter Plot")
plt.xlabel("Test Sample Index")
plt.ylabel("Confidence Score")
plt.tight_layout()
plt.savefig(f"{RESULTS_PATH}/scatter_plot.png", dpi=300)
plt.close()

print("\n✅ ALL Professional IEEE Graphs Generated Successfully!")
