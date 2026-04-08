import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Dataset
normal_path = "/content/drive/MyDrive/Machine_Fault_Detection/normal"
fault_path = "/content/drive/MyDrive/Machine_Fault_Detection/fault"

X = []
Y = []

for f in os.listdir(normal_path):
    X.append(extract_features(os.path.join(normal_path, f)))
    Y.append(0)

for f in os.listdir(fault_path):
    X.append(extract_features(os.path.join(fault_path, f)))
    Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

# Train
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Test input
test_file = input("Enter file name: ")
path = "/content/drive/MyDrive/" + test_file

y, sr = librosa.load(path, sr=None)

features = extract_features(path).reshape(1, -1)
prediction = model.predict(features)

if prediction[0] == 0:
    print("NORMAL")
else:
    print("FAULT")

# Waveform
plt.figure(figsize=(12,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of " + test_file)
plt.show()
