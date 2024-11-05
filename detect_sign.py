import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.camera_utils import initialize_camera, release_camera
from real_time_preprocessing import preprocess_frame  # Import the updated preprocessing function

def detect_sign_from_frame(model, frame):
    """Predict sign language character from a webcam frame."""
    preprocessed_frame = preprocess_frame(frame)  # Use the new preprocessing function
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction)
    return predicted_class

def start_real_time_detection(model_path):
    """Start real-time sign language detection using the webcam."""
    model = load_model(model_path)
    cap = initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        predicted_class = detect_sign_from_frame(model, frame)
        cv2.putText(frame, f"Predicted Sign: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(cap)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:/Users/Mohin/OneDrive/Desktop/VSCode/project/model/trained_model.h5"  # Adjust path as necessary
    start_real_time_detection(model_path)
