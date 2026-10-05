"""
Modern Desktop GUI for YOLOv3 Object Detection with Voice Feedback
Built using Tkinter + OpenCV + VoiceEngine
Features:
- Live Webcam, Image, and Video file processing
- Real-time video preview inside the GUI window
- Live adjustments for Confidence, NMS, and Voice Cooldown
- Choice of TTS Engine: Windows Native SAPI5 (Offline) or Google gTTS (Online)
- Detected objects log with confidence percentages and timestamps
- One-click Snapshot capture and disk saving
"""

import os
import sys
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from ObjectDetection import listBoundingBoxes, labelsBoundingBoxes, saveSnapshot
from voice_engine import VoiceEngine


class ModernObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionVoice - Real-Time YOLOv3 Object Detection")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#181920")

        # Application state
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.current_source_type = "webcam"  # 'webcam', 'image', 'video'
        self.selected_file_path = None
        
        # Load YOLO model
        self.status_text = tk.StringVar(value="Initializing YOLOv3 model...")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.count_var = tk.StringVar(value="Objects: 0")
        self.speech_status_var = tk.StringVar(value="Engine: SAPI (Offline)")
        
        # UI Control variables
        self.conf_val = tk.DoubleVar(value=0.50)
        self.nms_val = tk.DoubleVar(value=0.30)
        self.cooldown_val = tk.DoubleVar(value=4.0)
        self.voice_enabled_var = tk.BooleanVar(value=True)
        self.voice_engine_var = tk.StringVar(value="sapi")
        self.cam_index_var = tk.IntVar(value=0)

        # Threading & Detection state
        self.detection_lock = threading.Lock()
        self.latest_boxes = []
        self.latest_confs = []
        self.latest_class_ids = []
        self.latest_ids = ()
        self.latest_detected_classes = []
        self.frame_to_detect = None
        self.last_frame = None

        # Build interface
        self._build_styles()
        self._build_ui()
        
        # Initialize YOLO and Voice Engine in background thread to keep UI snappy
        threading.Thread(target=self._init_models, daemon=True).start()
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Color palette
        bg_dark = "#181920"
        card_bg = "#232530"
        accent_blue = "#3a86ff"
        text_light = "#f0f2f5"

        style.configure(".", background=bg_dark, foreground=text_light, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=card_bg, relief="flat")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13), background=card_bg, foreground="#ffffff")
        style.configure("Sub.TLabel", font=("Segoe UI", 9), background=card_bg, foreground="#a0a5b5")
        style.configure("Stat.TLabel", font=("Segoe UI Bold", 11), background=card_bg, foreground=accent_blue)
        style.configure("Accent.TButton", font=("Segoe UI Bold", 10), background=accent_blue, foreground="#ffffff", padding=6)
        style.map("Accent.TButton", background=[("active", "#2563eb"), ("disabled", "#374151")])
        style.configure("Secondary.TButton", font=("Segoe UI", 9), background="#333745", foreground="#ffffff", padding=5)
        style.map("Secondary.TButton", background=[("active", "#404656")])

    def _build_ui(self):
        # Main layout: Left = Video Stream & HUD, Right = Controls & Logs
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#181920", sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ----------------- LEFT PANEL (VIDEO PREVIEW) -----------------
        left_frame = ttk.Frame(main_paned, style="Card.TFrame")
        main_paned.add(left_frame, minsize=680)

        # Header bar in left panel
        left_header = ttk.Frame(left_frame, style="Card.TFrame")
        left_header.pack(fill=tk.X, padx=14, pady=10)

        title_lbl = ttk.Label(left_header, text="Live Detection Stream", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        stats_frame = ttk.Frame(left_header, style="Card.TFrame")
        stats_frame.pack(side=tk.RIGHT)

        ttk.Label(stats_frame, textvariable=self.fps_var, style="Stat.TLabel").pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, textvariable=self.count_var, style="Stat.TLabel").pack(side=tk.LEFT, padx=10)

        # Canvas for video rendering
        self.video_canvas = tk.Canvas(left_frame, bg="#0d0e12", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        # Bottom Stream Control Bar
        stream_ctrl = ttk.Frame(left_frame, style="Card.TFrame")
        stream_ctrl.pack(fill=tk.X, padx=14, pady=(0, 12))

        self.btn_start = ttk.Button(stream_ctrl, text="Start Webcam", style="Accent.TButton", command=self.toggle_stream)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_pause = ttk.Button(stream_ctrl, text="Pause", style="Secondary.TButton", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_snapshot = ttk.Button(stream_ctrl, text="Save Snapshot", style="Secondary.TButton", command=self.take_snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, padx=4)

        status_lbl = ttk.Label(stream_ctrl, textvariable=self.status_text, style="Sub.TLabel")
        status_lbl.pack(side=tk.RIGHT, padx=10)

        # ----------------- RIGHT PANEL (SETTINGS & LOGS) -----------------
        right_frame = ttk.Frame(main_paned, style="Card.TFrame")
        main_paned.add(right_frame, minsize=320)

        # Notebook (Tabs) for Settings and Detections
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tab_controls = ttk.Frame(notebook, style="Card.TFrame")
        tab_logs = ttk.Frame(notebook, style="Card.TFrame")
        notebook.add(tab_controls, text="  Controls  ")
        notebook.add(tab_logs, text="  Detections  ")

        # --- Controls Tab Content ---
        # 1. Input Source Selection
        src_group = tk.LabelFrame(tab_controls, text=" Input Source ", bg="#232530", fg="#8a99ad", font=("Segoe UI Semibold", 9))
        src_group.pack(fill=tk.X, padx=10, pady=8)

        src_btn_frame = ttk.Frame(src_group, style="Card.TFrame")
        src_btn_frame.pack(fill=tk.X, padx=6, pady=6)

        ttk.Button(src_btn_frame, text="Webcam", style="Secondary.TButton", command=self.select_webcam_mode).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(src_btn_frame, text="Browse Image", style="Secondary.TButton", command=self.select_image_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(src_btn_frame, text="Browse Video", style="Secondary.TButton", command=self.select_video_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 2. Voice Settings Group
        voice_group = tk.LabelFrame(tab_controls, text=" Voice & Speech Feedback ", bg="#232530", fg="#8a99ad", font=("Segoe UI Semibold", 9))
        voice_group.pack(fill=tk.X, padx=10, pady=8)

        chk_voice = tk.Checkbutton(
            voice_group, text="Enable Voice Announcements", variable=self.voice_enabled_var,
            bg="#232530", fg="#ffffff", selectcolor="#181920", activebackground="#232530",
            font=("Segoe UI", 9), command=self.on_voice_toggle
        )
        chk_voice.pack(anchor=tk.W, padx=8, pady=(4, 2))

        engine_frame = ttk.Frame(voice_group, style="Card.TFrame")
        engine_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(engine_frame, text="TTS Engine:", style="Sub.TLabel").pack(side=tk.LEFT)
        rb_sapi = tk.Radiobutton(
            engine_frame, text="SAPI5 (Offline)", variable=self.voice_engine_var, value="sapi",
            bg="#232530", fg="#ffffff", selectcolor="#181920", activebackground="#232530",
            command=self.on_engine_change
        )
        rb_sapi.pack(side=tk.LEFT, padx=6)
        rb_gtts = tk.Radiobutton(
            engine_frame, text="gTTS (Online)", variable=self.voice_engine_var, value="gtts",
            bg="#232530", fg="#ffffff", selectcolor="#181920", activebackground="#232530",
            command=self.on_engine_change
        )
        rb_gtts.pack(side=tk.LEFT, padx=6)

        cooldown_frame = ttk.Frame(voice_group, style="Card.TFrame")
        cooldown_frame.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(cooldown_frame, text="Announcement Cooldown (sec):", style="Sub.TLabel").pack(anchor=tk.W)
        self.cooldown_scale = tk.Scale(
            cooldown_frame, from_=1.0, to=10.0, resolution=0.5, orient=tk.HORIZONTAL,
            variable=self.cooldown_val, bg="#232530", fg="#ffffff", highlightthickness=0,
            troughcolor="#181920", command=self.on_cooldown_change
        )
        self.cooldown_scale.pack(fill=tk.X)

        # 3. Model Parameters Group
        model_group = tk.LabelFrame(tab_controls, text=" YOLOv3 Detection Settings ", bg="#232530", fg="#8a99ad", font=("Segoe UI Semibold", 9))
        model_group.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(model_group, text="Confidence Threshold:", style="Sub.TLabel").pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.conf_scale = tk.Scale(
            model_group, from_=0.10, to=0.95, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.conf_val, bg="#232530", fg="#ffffff", highlightthickness=0, troughcolor="#181920"
        )
        self.conf_scale.pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(model_group, text="NMS Suppression Threshold:", style="Sub.TLabel").pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.nms_scale = tk.Scale(
            model_group, from_=0.10, to=0.90, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.nms_val, bg="#232530", fg="#ffffff", highlightthickness=0, troughcolor="#181920"
        )
        self.nms_scale.pack(fill=tk.X, padx=8, pady=2)

        # --- Detections Tab Content ---
        tree_frame = ttk.Frame(tab_logs, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = ("time", "class", "confidence")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=14)
        self.tree.heading("time", text="Time")
        self.tree.heading("class", text="Object")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("time", width=70, anchor=tk.CENTER)
        self.tree.column("class", width=120, anchor=tk.W)
        self.tree.column("confidence", width=90, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(tab_logs, text="Clear Detection History", style="Secondary.TButton", command=self.clear_logs).pack(fill=tk.X, padx=6, pady=4)

    def _init_models(self):
        """Load YOLOv3 model and VoiceEngine in the background."""
        try:
            model_cfg = 'model/yolov3.cfg'
            model_weights = 'model/yolov3.weights'
            labels_path = 'model/yolov3-labels'

            if not os.path.exists(labels_path) or not os.path.exists(model_cfg) or not os.path.exists(model_weights):
                self.status_text.set("Error: Model files missing in model/ directory.")
                return

            with open(labels_path, 'r') as f:
                self.class_labels = [line.strip() for line in f if line.strip()]

            np.random.seed(42)
            self.label_colors = np.random.randint(40, 240, size=(len(self.class_labels), 3), dtype='uint8')

            self.net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

            layer_names = self.net.getLayerNames()
            unconn = self.net.getUnconnectedOutLayers()
            if hasattr(unconn, 'flatten'):
                unconn = unconn.flatten()
            self.output_layers = [layer_names[int(i) - 1] for i in unconn]

            # Voice Engine
            self.voice = VoiceEngine(
                engine_type=self.voice_engine_var.get(),
                cooldown_seconds=self.cooldown_val.get()
            )

            self.status_text.set("Ready. Click 'Start Webcam' or browse a file.")
        except Exception as e:
            self.status_text.set(f"Init Error: {str(e)}")

    def select_webcam_mode(self):
        self.stop_current_stream()
        self.current_source_type = "webcam"
        self.status_text.set("Selected: Webcam mode. Click 'Start Webcam'.")
        self.btn_start.configure(text="▶ Start Webcam")

    def select_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self.stop_current_stream()
            self.current_source_type = "image"
            self.selected_file_path = file_path
            self.status_text.set(f"Loaded image: {os.path.basename(file_path)}")
            self.process_static_image(file_path)

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")]
        )
        if file_path:
            self.stop_current_stream()
            self.current_source_type = "video"
            self.selected_file_path = file_path
            self.status_text.set(f"Loaded video: {os.path.basename(file_path)}")
            self.btn_start.configure(text="▶ Play Video")

    def toggle_stream(self):
        if not self.is_running:
            if self.current_source_type == "webcam":
                self.start_webcam_stream()
            elif self.current_source_type == "video":
                if self.selected_file_path:
                    self.start_video_stream(self.selected_file_path)
                else:
                    self.select_video_file()
        else:
            self.stop_current_stream()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.configure(text="▶ Resume")
            self.status_text.set("Stream paused.")
        else:
            self.btn_pause.configure(text="⏸ Pause")
            self.status_text.set("Stream running.")

    def start_webcam_stream(self):
        self.cap = cv.VideoCapture(self.cam_index_var.get())
        if not self.cap.isOpened():
            self.cap = cv.VideoCapture(1)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open webcam. Check camera connection.")
                return

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        self.is_paused = False
        self.btn_start.configure(text="⏹ Stop Stream")
        self.btn_pause.configure(state=tk.NORMAL)
        self.status_text.set("Webcam streaming live.")

        # Start async detection worker
        threading.Thread(target=self._async_detection_worker, daemon=True).start()
        # Start display loop
        self.root.after(10, self._stream_update_loop)

    def start_video_stream(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video file: {video_path}")
            return

        self.is_running = True
        self.is_paused = False
        self.btn_start.configure(text="⏹ Stop Video")
        self.btn_pause.configure(state=tk.NORMAL)
        self.status_text.set("Playing video file.")

        threading.Thread(target=self._async_detection_worker, daemon=True).start()
        self.root.after(10, self._stream_update_loop)

    def stop_current_stream(self):
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.configure(text="▶ Start")
        self.btn_pause.configure(state=tk.DISABLED, text="⏸ Pause")
        self.status_text.set("Stream stopped.")
        self.fps_var.set("FPS: --")

    def _async_detection_worker(self):
        """Runs YOLO forward pass in background without stalling GUI or video."""
        while self.is_running:
            frame_copy = None
            with self.detection_lock:
                if self.frame_to_detect is not None:
                    frame_copy = self.frame_to_detect.copy()
                    self.frame_to_detect = None

            if frame_copy is None:
                time.sleep(0.01)
                continue

            try:
                h, w = frame_copy.shape[:2]
                blob = cv.dnn.blobFromImage(frame_copy, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)

                conf_th = self.conf_val.get()
                nms_th = self.nms_val.get()
                boxes, confs, cids = listBoundingBoxes(outs, h, w, conf_th)

                ids = ()
                if len(boxes) > 0:
                    ids = cv.dnn.NMSBoxes(boxes, confs, conf_th, nms_th)

                classes = []
                log_entries = []
                if ids is not None and len(ids) > 0:
                    flat_ids = np.array(ids).flatten()
                    now_str = datetime.now().strftime("%H:%M:%S")
                    for i in flat_ids:
                        if i < len(cids):
                            cname = self.class_labels[cids[i]]
                            classes.append(cname)
                            log_entries.append((now_str, cname, f"{int(confs[i]*100)}%"))

                with self.detection_lock:
                    self.latest_boxes = boxes
                    self.latest_confs = confs
                    self.latest_class_ids = cids
                    self.latest_ids = ids
                    self.latest_detected_classes = classes

                # Update Detections table in UI thread
                if log_entries:
                    self.root.after(0, self._add_log_entries, log_entries)

                # Voice Announcement
                if classes and self.voice_enabled_var.get() and hasattr(self, 'voice'):
                    self.voice.announce_objects(classes)

            except Exception as e:
                pass

    def _stream_update_loop(self):
        """Updates the video frame on the Tkinter canvas."""
        if not self.is_running or not self.cap:
            return

        if not self.is_paused:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.current_source_type == "video":
                    self.stop_current_stream()
                    self.status_text.set("Video finished.")
                return

            with self.detection_lock:
                if self.frame_to_detect is None:
                    self.frame_to_detect = frame

            with self.detection_lock:
                boxes = list(self.latest_boxes)
                confs = list(self.latest_confs)
                cids = list(self.latest_class_ids)
                ids = self.latest_ids
                classes = list(self.latest_detected_classes)

            annotated_frame, _ = labelsBoundingBoxes(
                frame.copy(), boxes, confs, cids, ids, self.label_colors, self.class_labels
            )
            self.last_frame = annotated_frame
            self.count_var.set(f"Objects: {len(classes)}")

            # Render to canvas
            self._render_frame_to_canvas(annotated_frame)

        self.root.after(20, self._stream_update_loop)

    def process_static_image(self, image_path):
        """Process and display a still image."""
        try:
            image = cv.imread(image_path)
            if image is None:
                messagebox.showerror("Error", f"Failed to load image: {image_path}")
                return

            h, w = image.shape[:2]
            blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            conf_th = self.conf_val.get()
            nms_th = self.nms_val.get()
            boxes, confs, cids = listBoundingBoxes(outs, h, w, conf_th)

            ids = ()
            if len(boxes) > 0:
                ids = cv.dnn.NMSBoxes(boxes, confs, conf_th, nms_th)

            annotated_image, detected_classes = labelsBoundingBoxes(
                image.copy(), boxes, confs, cids, ids, self.label_colors, self.class_labels
            )

            self.last_frame = annotated_image
            self.count_var.set(f"Objects: {len(detected_classes)}")
            self._render_frame_to_canvas(annotated_image)

            # Log entries
            now_str = datetime.now().strftime("%H:%M:%S")
            log_entries = []
            if ids is not None and len(ids) > 0:
                flat_ids = np.array(ids).flatten()
                for i in flat_ids:
                    if i < len(cids):
                        cname = self.class_labels[cids[i]]
                        log_entries.append((now_str, cname, f"{int(confs[i]*100)}%"))
            self._add_log_entries(log_entries)

            # Voice announcement
            if detected_classes and self.voice_enabled_var.get() and hasattr(self, 'voice'):
                text = self.voice.format_speech_text(detected_classes)
                self.voice.say(text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed processing image: {str(e)}")

    def _render_frame_to_canvas(self, cv_image):
        """Fit and render OpenCV BGR image onto Tkinter canvas."""
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            return

        h, w = cv_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv.resize(cv_image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        rgb_image = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)
        self.tk_photo = ImageTk.PhotoImage(image=pil_img)

        self.video_canvas.delete("all")
        # Center image
        pos_x = (canvas_width - new_w) // 2
        pos_y = (canvas_height - new_h) // 2
        self.video_canvas.create_image(pos_x, pos_y, anchor=tk.NW, image=self.tk_photo)

    def _add_log_entries(self, entries):
        """Add new detections to top of treeview."""
        for entry in entries:
            self.tree.insert("", 0, values=entry)
        # Keep tree bounded to 100 entries max
        children = self.tree.get_children()
        if len(children) > 100:
            for item in children[100:]:
                self.tree.delete(item)

    def clear_logs(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def take_snapshot(self):
        if self.last_frame is not None:
            saved_path = saveSnapshot(self.last_frame)
            self.status_text.set(f"Snapshot saved to: {os.path.basename(saved_path)}")
            messagebox.showinfo("Snapshot Saved", f"Frame saved successfully to:\n{saved_path}")
        else:
            messagebox.showwarning("Snapshot", "No active frame to save.")

    def on_voice_toggle(self):
        if hasattr(self, 'voice'):
            self.voice.muted = not self.voice_enabled_var.get()
            status = "Enabled" if self.voice_enabled_var.get() else "Muted"
            self.status_text.set(f"Voice {status}.")

    def on_engine_change(self):
        engine = self.voice_engine_var.get()
        if hasattr(self, 'voice'):
            self.voice.engine_type = engine
            self.status_text.set(f"Voice engine set to: {engine.upper()}")

    def on_cooldown_change(self, val):
        if hasattr(self, 'voice'):
            self.voice.cooldown_seconds = float(val)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'voice'):
            self.voice.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModernObjectDetectionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
