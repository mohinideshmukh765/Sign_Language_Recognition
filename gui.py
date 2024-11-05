import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from src.utils.camera_utils import initialize_camera, release_camera
from src.real_time_preprocessing import preprocess_frame

# Load your trained model
model = load_model("C:/Users/Mohin/OneDrive/Desktop/VSCode/project/model/trained_model.h5")

class_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
               10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 
               19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "0", 27: "1", 
               28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9"}

def predict_sign_from_frame(frame):
    """Process and predict the sign language character from the frame."""
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = prediction.argmax()
    return class_names.get(predicted_class, "Unknown")

def start_camera():
    """Start the webcam feed and display real-time sign language predictions."""
    cap = initialize_camera()

    if not cap.isOpened():
        tk.messagebox.showerror("Camera Error", "Failed to open the camera")
        return

    # Create main camera feed window
    camera_window = tk.Toplevel()
    camera_window.title("Camera Feed")
    camera_window.geometry("800x600")

    # Label to show camera feed
    label = tk.Label(camera_window)
    label.pack()

    # Stop camera 
    def stop_camera():
        release_camera(cap)
        camera_window.destroy()

    # Stop button
    stop_button = tk.Button(camera_window, text="Stop Camera", command=stop_camera)
    stop_button.pack(pady=20)

    # Function to update camera feed and predict sign language dynamically
    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Predict the sign language character
            detected_character = predict_sign_from_frame(frame)

            # Add text (predicted character) to the frame
            cv2.putText(frame, f"Character = {detected_character}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert the frame to a format suitable for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk  # Keep a reference to avoid garbage collection
            label.configure(image=imgtk)

        camera_window.after(10, update_frame)  # Update frame every 10 ms

    update_frame()  # Start updating frames

    camera_window.protocol("WM_DELETE_WINDOW", stop_camera)  # Handle window close
