
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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

df = pd.read_csv("dataset.csv")

X, y, groups = [], [], []
for _, row in df.iterrows():
    X.append(extract_logmel_sequence(row["file_path"]))
    y.append(row["label"])
    groups.append(row["speaker"])

X = np.array(X)
y = to_categorical(y)

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = Sequential([
    Masking(mask_value=0., input_shape=(300, 64)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early = EarlyStopping(monitor='val_accuracy',
                      patience=5,
                      restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print("\nFinal BiLSTM Test Accuracy:", acc)

model.save("../models/bilstm_speakerwise.keras")  

with open("../results/bilstm_speakerwise.txt", "w") as f:
    f.write(f"Accuracy: {acc}")

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('BiLSTM Training Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../results/bilstm_training_curve.png')
plt.close()
