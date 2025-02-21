import numpy as np
import imutils
import pickle
import time
import cv2

# File paths
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5  # Confidence threshold

# Loading face detector
print("Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Loading face recognizer
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Starting video stream
print("Starting video stream...")
cam = cv2.VideoCapture(0)  # Use index 1 or 2 if index 0 doesn't work
time.sleep(2.0)  # Allow time for camera to initialize

if not cam.isOpened():
    print("Error: Camera not detected or inaccessible.")
    exit()

while True:
    ret, frame = cam.read()
    
    # Check if frame was successfully captured
    if not ret or frame is None:
        print("Error: Failed to grab frame. Check your camera connection.")
        break

    # Resize the frame to 600px width for faster processing
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Detect faces in the frame
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only consider detections above the confidence threshold
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face from the frame
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Skip small faces that may not be detected properly
            if fW < 20 or fH < 20:
                continue

            # Create a face blob for face recognition
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict the person's identity
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Display the predicted name and confidence on the frame
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'ESC' to exit
    if key == 27:
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
