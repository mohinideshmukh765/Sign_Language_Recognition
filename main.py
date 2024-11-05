import os
# Disable GPU and force TensorFlow to use only CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tkinter as tk
from gui import start_camera

def main():
    root = tk.Tk()
    root.title("Sign Language Detection")
    root.geometry("800x600")

    start_button = tk.Button(root, text="Start Camera", command=start_camera)
    start_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
