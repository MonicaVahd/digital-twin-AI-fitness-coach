import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from io import StringIO

def load_opensmile_vector(file_path):
    """
    Loads a feature vector from an OpenSmile ARFF/CSV file.
    Only uses the numeric columns after the @data line.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("@data not found in file")
    data_str = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(data_str), header=None)
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    vector = df.values[0].astype(float)
    return vector

def predict_emotion_from_csv(csv_path, reference_dir='reference_vectors'):
    """
    Compares a real-time feature vector to all reference vectors and returns the predicted emotion.
    """
    realtime_vector = load_opensmile_vector(csv_path)
    print("Real-time vector shape:", realtime_vector.shape)
    reference_vectors = {}
    for fname in os.listdir(reference_dir):
        if fname.endswith('.npy'):
            emotion = fname.replace('_reference.npy', '')
            reference_vectors[emotion] = np.load(os.path.join(reference_dir, fname))
            print(f"{emotion} reference vector shape: {reference_vectors[emotion].shape}")
    scores = {}
    for emotion, ref_vec in reference_vectors.items():
        score = cosine_similarity([realtime_vector], [ref_vec])[0][0]
        scores[emotion] = score
    predicted_emotion = max(scores, key=scores.get)
    return predicted_emotion, scores

if __name__ == "__main__":
    # Example usage:
    # Suppose you have a real-time OpenSmile output at 'output.csv'
    # and your reference vectors are in 'reference_vectors/'
    csv_path = 'output.csv'  # Change this to your real-time feature file
    reference_dir = 'reference_vectors'
    try:
        predicted_emotion, scores = predict_emotion_from_csv(csv_path, reference_dir)
        print("Predicted emotion:", predicted_emotion)
        print("Similarity scores:", scores)
    except Exception as e:
        print(f"Error: {e}")