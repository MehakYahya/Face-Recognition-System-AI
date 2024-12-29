import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess_data import preprocess_data

# Constants
MODEL_PATH = "../results/face_recognition_model.h5"
ATTENDANCE_FILE = "../results/attendance.csv"
IMG_SIZE = 128
DATASET_PATH = r"C:\Users\dell\Downloads\PythonProject1\Student attendance.v1i.retinanet"

def log_attendance(name):
    """
    Logs the attendance of a recognized individual.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.DataFrame(columns=["Name", "Time"])
    else:
        attendance_df = pd.read_csv(ATTENDANCE_FILE)

    if name not in attendance_df["Name"].values:
        new_entry = pd.DataFrame({"Name": [name], "Time": [pd.Timestamp.now()]})
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)

def main():
    """
    Real-time face recognition and attendance logging using a trained model.
    """
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Open webcam for real-time recognition
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time face recognition. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Preprocess the frame
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img / 255.0, axis=0)

        # Predict the class
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=-1)  # For multi-class
        name = f"Student {predicted_class[0]}"  # Assuming label_map uses "Student 1", "Student 2", etc.

        # Display the name on the frame
        cv2.putText(frame, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)

        # Log attendance
        log_attendance(name)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
