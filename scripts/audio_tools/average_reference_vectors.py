import os
import pandas as pd
import numpy as np
from io import StringIO

FEATURES_ROOT = 'opensmile_features'
REFERENCE_DIR = 'reference_vectors'
os.makedirs(REFERENCE_DIR, exist_ok=True)

for emotion in os.listdir(FEATURES_ROOT):
    emotion_dir = os.path.join(FEATURES_ROOT, emotion)
    if not os.path.isdir(emotion_dir):
        continue
    all_vectors = []
    for root, dirs, files in os.walk(emotion_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    data_start = None
                    for i, line in enumerate(lines):
                        if line.strip().lower() == '@data':
                            data_start = i + 1
                            break
                    if data_start is None:
                        print(f"@data not found in {file_path}, skipping.")
                        continue
                    data_str = ''.join(lines[data_start:])
                    df = pd.read_csv(StringIO(data_str), header=None)
                    # Keep only numeric columns
                    df = df.select_dtypes(include=[np.number])
                    # Convert to float and flatten
                    vector = df.values[0].astype(float)
                    all_vectors.append(vector)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    if all_vectors:
        avg_vector = np.mean(all_vectors, axis=0)
        np.save(os.path.join(REFERENCE_DIR, f'{emotion}_reference.npy'), avg_vector)
        print(f"Saved reference vector for {emotion} ({len(all_vectors)} samples)")
    else:
        print(f"No valid feature vectors found for {emotion}")