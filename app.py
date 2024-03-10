import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('ASLNEW.h5')

# Initialize a video capture object
cap = cv2.VideoCapture(0)

# Define the list of labels that the model can predict
labels = ['A', 'B', 'C', 'D', 'E']

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame to match the input shape and type expected by the model
    img = cv2.resize(frame, (64, 64))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)

    # Use the model to make a prediction on the preprocessed frame
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()