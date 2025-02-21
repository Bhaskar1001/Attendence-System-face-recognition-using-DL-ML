from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# Initializing paths
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Ensure the embedding file exists
if not os.path.exists(embeddingFile):
    raise FileNotFoundError(f"Embedding file '{embeddingFile}' not found.")

print("Loading face embeddings...")
with open(embeddingFile, "rb") as f:
    data = pickle.load(f)

# Check for unique names
unique_names = set(data["names"])
print(f"Unique names: {unique_names}")
print(f"Total unique names: {len(unique_names)}")

# Ensure there are multiple unique classes
if len(unique_names) <= 1:
    raise ValueError("Training data must contain more than one unique class.")

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Ensure the output directory exists
output_dir = os.path.dirname(recognizerFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Saving the trained model and label encoder
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)

with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)

print("Model and label encoder saved successfully.")

