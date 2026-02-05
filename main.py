import cv2
import threading
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO

# Optimization: Force faster camera backend for Windows
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

EXCEL_FILE = os.path.join(get_desktop_path(), "acne_history.xlsx")
IMAGE_SAVE_DIR = os.path.join(get_desktop_path(), "Acne_Annotated_Images")
MODEL_PATH = "best.pt"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AcneKioskApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Acne Detection Kiosk")
        self.geometry("1200x800") 
        self.attributes("-fullscreen", False)

        self.model = None
        self.preview_cap = None
        self.preview_running = False
        self.last_annotated_frame = None
        
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.build_ui()
        threading.Thread(target=self._load_model_background, daemon=True).start()
        self.load_history()

    def _load_model_background(self):
        loaded_model = YOLO(MODEL_PATH)
        loaded_model(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, verbose=False)
        self.model = loaded_model
        self.after(0, lambda: self.status_label.configure(text="Model Ready", text_color="#22c55e"))

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.top_frame = ctk.CTkFrame(self, height=80)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.patient_entry = ctk.CTkEntry(self.top_frame, placeholder_text="Enter Patient Name", width=400, height=45, font=("Arial", 18))
        self.patient_entry.pack(side="left", padx=20)
        
        self.status_label = ctk.CTkLabel(self.top_frame, text="Loading model...", font=("Arial", 14))
        self.status_label.pack(side="left", padx=20)
        
        self.acne_frame = ctk.CTkFrame(self.top_frame, corner_radius=12)
        self.acne_frame.pack(side="right", padx=30)
        ctk.CTkLabel(self.acne_frame, text="TOTAL ACNE", font=("Arial", 14, "bold")).pack(padx=20, pady=(10, 0))
        self.acne_count_label = ctk.CTkLabel(self.acne_frame, text="0", font=("Arial", 40, "bold"), text_color="#22c55e")
        self.acne_count_label.pack(padx=20, pady=(0, 10))
        
        self.center_frame = ctk.CTkFrame(self)
        self.center_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.center_frame, text="")
        self.image_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkButton(self.btn_frame, text="Open Camera", command=self.open_camera).pack(side="left", padx=20)
        ctk.CTkButton(self.btn_frame, text="Capture Image", command=self.capture_image).pack(side="left", padx=20)
        ctk.CTkButton(self.btn_frame, text="Save Image", command=self.save_annotated_image).pack(side="left", padx=20)
        ctk.CTkButton(self.btn_frame, text="Exit", fg_color="red", command=self.exit_app).pack(side="right", padx=20)
        
        self.sidebar = ctk.CTkFrame(self, width=280)
        self.sidebar.grid(row=1, column=1, rowspan=2, sticky="ns", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="History", font=("Arial", 18, "bold")).pack(pady=10)
        self.history_box = ctk.CTkTextbox(self.sidebar, width=260)
        self.history_box.pack(expand=True, fill="both", padx=10, pady=10)

    def open_camera(self):
        if self.preview_running: return
        self.preview_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # --- QUALITY IMPROVEMENT ---
        # Requesting HD resolution from the hardware
        self.preview_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.preview_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # ---------------------------
        
        self.preview_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        if not self.preview_cap.isOpened():
            self.status_label.configure(text="Camera not found")
            return
        
        self.preview_running = True
        self.status_label.configure(text="Camera preview (HD)")
        
        threading.Thread(target=self._update_frame_thread, daemon=True).start()
        self.update_preview()

    def _update_frame_thread(self):
        while self.preview_running:
            ret, frame = self.preview_cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            else:
                break

    def update_preview(self):
        if not self.preview_running: return
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
        
        if frame is not None:
            # Downscale only for the UI preview to keep the interface snappy
            preview_img = cv2.resize(frame, (800, 600))
            frame_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(img, size=(800, 600))
            self.image_label.configure(image=ctk_img)
            self.image_label.image = ctk_img
        
        self.after(10, self.update_preview)

    def close_camera(self):
        self.preview_running = False
        if self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None

    def capture_image(self):
        if not self.preview_running or self.model is None: return
        with self.frame_lock:
            high_res_frame = self.current_frame.copy() if self.current_frame is not None else None
        self.close_camera()
        if high_res_frame is None: return

        # --- SPEED OPTIMIZATION ---
        # The model sees a resized version, but we save the original HD version
        # This keeps the model results consistent and fast
        results = self.model.predict(high_res_frame, conf=0.12, imgsz=320, verbose=False)[0]
        # ---------------------------

        acne_count = len(results.boxes)
        self.acne_count_label.configure(text=str(acne_count))

        # Render boxes on the High-Res frame
        annotated_frame = results.plot() 
        acne_sizes = [f"{int(b.xyxy[0][2]-b.xyxy[0][0])}x{int(b.xyxy[0][3]-b.xyxy[0][1])}" for b in results.boxes]

        self.last_annotated_frame = annotated_frame.copy()
        
        # Display the result (resized to fit the UI label)
        display_res = cv2.resize(annotated_frame, (800, 600))
        annotated_rgb = cv2.cvtColor(display_res, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_rgb)
        ctk_img = ctk.CTkImage(img, size=(800, 600))
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img
        self.save_to_excel(acne_count, ", ".join(acne_sizes))

    def save_annotated_image(self):
        if self.last_annotated_frame is None:
            self.status_label.configure(text="No image to save")
            return
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        patient = self.patient_entry.get() or "Unknown"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{patient}_{timestamp}.jpg"
        # Saves in full camera resolution
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, filename), self.last_annotated_frame)
        self.status_label.configure(text="HD Image saved to Desktop")

    def save_to_excel(self, count, sizes):
        row = {"Date": datetime.now().strftime("%Y-%m-%d"), "Patient": self.patient_entry.get() or "Unknown", "Total Acne": count, "Sizes": sizes}
        if os.path.exists(EXCEL_FILE): df = pd.read_excel(EXCEL_FILE)
        else: df = pd.DataFrame(columns=row.keys())
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        self.load_history()

    def load_history(self):
        self.history_box.delete("1.0", "end")
        if not os.path.exists(EXCEL_FILE): return
        df = pd.read_excel(EXCEL_FILE)
        for _, r in df.iterrows():
            self.history_box.insert("end", f"{r['Date']} | {r['Patient']} | Acne: {r['Total Acne']}\n")

    def exit_app(self):
        self.close_camera()
        self.destroy()

if __name__ == "__main__":
    app = AcneKioskApp()
    app.mainloop()