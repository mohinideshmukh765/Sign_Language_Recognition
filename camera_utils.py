import cv2

def initialize_camera():
    # Use CAP_DSHOW as the backend on Windows for better compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Updated to use DirectShow backend
    if not cap.isOpened():
        raise Exception("Failed to open the camera")
    return cap

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
