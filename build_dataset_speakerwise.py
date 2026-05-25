import os
import pandas as pd

BASE = r"D:\cognitoTrack_Speech\data"
rows = []

for label_name, label in [("control", 0), ("dementia", 1)]:
    label_path = os.path.join(BASE, label_name)

    for speaker in os.listdir(label_path):
        speaker_path = os.path.join(label_path, speaker)

        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    rows.append([
                        os.path.join(speaker_path, file),
                        label,
                        speaker
                    ])

df = pd.DataFrame(rows, columns=["file_path", "label", "speaker"])
df.to_csv("dataset.csv", index=False)
print("dataset.csv created")
