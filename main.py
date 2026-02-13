import os
import sys
import threading
import cv2
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image
from datetime import datetime
import multiprocessing

# --- 1. PERFORMANCE & PACKAGING CONFIG ---
os.environ["SETTINGS_SYNC"] = "False"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

EXCEL_FILE = os.path.join(get_desktop_path(), "acne_history.xlsx")
IMAGE_SAVE_DIR = os.path.join(get_desktop_path(), "Acne_Annotated_Images")
MODEL_PATH = resource_path("best.pt")

class AcneKioskApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Acne Detection (PyTorch CPU)")
        self.geometry("1200x820")
        
        self.model = None
        self.preview_cap = None
        self.preview_running = False
        self.current_frame = None
        self.last_annotated_frame = None
        self.frame_lock = threading.Lock()

        self.build_ui()
        # Start loading background thread
        threading.Thread(target=self._lazy_load_engine, daemon=True).start()
        self.load_history()

    def _lazy_load_engine(self):
        try:
            global YOLO
            from ultralytics import YOLO
            
            # Check if model exists before loading
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

            loaded_model = YOLO(MODEL_PATH)
            # Warmup
            loaded_model(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, verbose=False)
            
            self.model = loaded_model
            # Schedule UI update on main thread
            self.after(0, self._set_status_ready)
        except Exception as err:
            # We convert err to string here so it persists in the lambda
            error_msg = str(err)
            self.after(0, lambda: self.status_label.configure(text=f"Error: {error_msg}", text_color="red"))

    def _set_status_ready(self):
        self.status_label.configure(text="● System Ready", text_color="#22c55e")

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.top_frame = ctk.CTkFrame(self, height=100, corner_radius=0)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        self.patient_entry = ctk.CTkEntry(self.top_frame, placeholder_text="Patient Name...", width=350, height=45)
        self.patient_entry.pack(side="left", padx=20, pady=20)
        
        self.status_label = ctk.CTkLabel(self.top_frame, text="● Initializing...", text_color="#f59e0b", font=("Arial", 13, "bold"))
        self.status_label.pack(side="left", padx=10)

        self.count_card = ctk.CTkFrame(self.top_frame, fg_color="#1e293b", width=150)
        self.count_card.pack(side="right", padx=20, pady=10)
        ctk.CTkLabel(self.count_card, text="ACNE COUNT", font=("Arial", 10)).pack(padx=10)
        self.acne_count_label = ctk.CTkLabel(self.count_card, text="0", font=("Arial", 32, "bold"), text_color="#3b82f6")
        self.acne_count_label.pack(padx=10, pady=(0, 5))

        # Main Display
        self.display_frame = ctk.CTkFrame(self, fg_color="#0f172a")
        self.display_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.display_frame, text="Camera Offline", font=("Arial", 20))
        self.image_label.pack(expand=True, fill="both")

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.grid(row=1, column=1, rowspan=2, sticky="ns", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="Recent History", font=("Arial", 16, "bold")).pack(pady=10)
        self.history_box = ctk.CTkTextbox(self.sidebar, width=280, font=("Consolas", 12))
        self.history_box.pack(expand=True, fill="both", padx=10, pady=10)

        # Buttons
        self.btn_frame = ctk.CTkFrame(self, height=80)
        self.btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkButton(self.btn_frame, text="Open Camera", command=self.open_camera, height=45).pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="Capture & Analyze", command=self.capture_image, fg_color="#2563eb", height=45).pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="Save Record", command=self.save_annotated_image, height=45).pack(side="left", padx=10)
        ctk.CTkButton(self.btn_frame, text="Exit", fg_color="#475569", command=self.exit_app, height=45).pack(side="right", padx=10)

    def open_camera(self):
        if self.preview_running: return
        self.preview_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.preview_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.preview_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.preview_cap.isOpened():
            self.status_label.configure(text="● Camera Error", text_color="red")
            return
        
        self.preview_running = True
        threading.Thread(target=self._stream_worker, daemon=True).start()
        self._update_ui_frame()

    def _stream_worker(self):
        while self.preview_running:
            ret, frame = self.preview_cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            else: break

    def _update_ui_frame(self):
        if not self.preview_running: return
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
        
        if frame is not None:
            preview = cv2.cvtColor(cv2.resize(frame, (850, 500)), cv2.COLOR_BGR2RGB)
            img = ctk.CTkImage(Image.fromarray(preview), size=(850, 500))
            self.image_label.configure(image=img, text="")
        
        self.after(15, self._update_ui_frame)

    def capture_image(self):
        if not self.preview_running or self.model is None: return
        
        with self.frame_lock:
            if self.current_frame is None: return
            snap = self.current_frame.copy()
        
        self.preview_running = False
        if self.preview_cap: self.preview_cap.release()
        
        # In .pt mode, 320 is safe and efficient
        results = self.model.predict(snap, conf=0.25, imgsz=320, verbose=False)[0]
        
        self.last_annotated_frame = results.plot()
        self.acne_count_label.configure(text=str(len(results.boxes)))
        
        res_rgb = cv2.cvtColor(cv2.resize(self.last_annotated_frame, (850, 500)), cv2.COLOR_BGR2RGB)
        img = ctk.CTkImage(Image.fromarray(res_rgb), size=(850, 500))
        self.image_label.configure(image=img)
        
        self.save_to_excel(len(results.boxes))

    def save_annotated_image(self):
        if self.last_annotated_frame is None: return
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        name = self.patient_entry.get() or "Unknown"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, f"{name}_{ts}.jpg"), self.last_annotated_frame)
        self.status_label.configure(text=f"● Saved {ts}", text_color="#22c55e")

    def save_to_excel(self, count):
        data = {"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), 
                "Patient": self.patient_entry.get() or "Unknown", 
                "Count": count}
        try:
            df = pd.read_excel(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_excel(EXCEL_FILE, index=False)
        except Exception: pass
        self.load_history()

    def load_history(self):
        self.history_box.delete("1.0", "end")
        if not os.path.exists(EXCEL_FILE): return
        try:
            df = pd.read_excel(EXCEL_FILE).tail(15)
            for _, r in df.iloc[::-1].iterrows():
                self.history_box.insert("end", f"{r['Date']}\n{r['Patient']}: {r['Count']} spots\n{'-'*20}\n")
        except Exception: pass

    def exit_app(self):
        self.preview_running = False
        if self.preview_cap: self.preview_cap.release()
        self.destroy()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = AcneKioskApp()
    app.mainloop()