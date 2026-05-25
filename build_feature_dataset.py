import pandas as pd
import time
from feature_extraction import extract_features

df = pd.read_csv("dataset.csv")

rows = []
total = len(df)

start_time = time.time()

print(f"\nProcessing {total} audio files...\n")

for index, row in df.iterrows():

    features = extract_features(row["file_path"])

    features["label"] = row["label"]
    features["speaker"] = row["speaker"]

    rows.append(features)

    if (index + 1) % 10 == 0:
        print(f"Processed {index+1}/{total}")

feature_df = pd.DataFrame(rows)
feature_df.to_csv("features_dataset_named.csv", index=False)

end_time = time.time()

print("\n✅ Feature extraction completed")
print("⏱ Total time:", round((end_time - start_time)/60, 2), "minutes")
