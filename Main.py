"""
Main Application for Real-Time YOLOv3 Object Detection with Voice Feedback
Features:
- Dual-threaded architecture: 30+ FPS smooth video feed with async YOLO inference
- Voice Engine integration (Offline Windows SAPI5 + Online Google gTTS)
- Heads-Up Display (HUD) overlay with live FPS, object counts, and audio status
- Full hotkey support:
    [Q] / [ESC]  : Quit
    [M] / [V]    : Toggle Voice Mute / Unmute
    [E]          : Toggle TTS Engine (SAPI <-> gTTS)
    [S]          : Save Annotated Snapshot
    [SPACE]      : Pause / Resume Stream
    [+] / [-]    : Adjust Confidence Threshold
- Support for Webcam, Video files, Image files, and GUI launch (--gui)
"""

import sys
import os
import time
import argparse
import threading
import numpy as np
import cv2 as cv

from ObjectDetection import (
    detectObject,
    drawHUD,
    saveSnapshot,
    displayImage,
    listBoundingBoxes,
    labelsBoundingBoxes
)
from voice_engine import VoiceEngine


class YOLOVoiceApp:
    def __init__(self,
                 model_cfg='model/yolov3.cfg',
                 model_weights='model/yolov3.weights',
                 labels_path='model/yolov3-labels',
                 conf_threshold=0.5,
                 nms_threshold=0.3,
                 voice_engine_type='sapi',
                 enable_voice=True,
                 voice_cooldown=4.0):
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.enable_voice = enable_voice
        self.is_paused = False
        self.is_running = False
        
        # Load labels and model
        print("[INFO] Loading YOLOv3 model and labels...")
        if not os.path.exists(model_cfg) or not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model files not found. Ensure {model_cfg} and {model_weights} exist.")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
        with open(labels_path, 'r') as f:
            self.class_labels = [line.strip() for line in f if line.strip()]
            
        # Assign consistent, vibrant colors to classes
        np.random.seed(42)
        self.label_colors = np.random.randint(40, 240, size=(len(self.class_labels), 3), dtype='uint8')
        
        # Initialize OpenCV DNN Darknet Network
        self.net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        
        # Retrieve unconnected output layer names
        layer_names = self.net.getLayerNames()
        unconn = self.net.getUnconnectedOutLayers()
        if hasattr(unconn, 'flatten'):
            unconn = unconn.flatten()
        self.output_layers = [layer_names[int(i) - 1] for i in unconn]
        
        print(f"[INFO] YOLOv3 loaded successfully with {len(self.class_labels)} classes.")
        
        # Initialize Voice Engine
        self.voice = VoiceEngine(
            engine_type=voice_engine_type,
            cooldown_seconds=voice_cooldown,
            sound_dir='play'
        )
        if not enable_voice:
            self.voice.muted = True
            
        # Shared detection state for async pipeline
        self.latest_boxes = []
        self.latest_confs = []
        self.latest_class_ids = []
        self.latest_ids = ()
        self.latest_detected_classes = []
        self.detection_lock = threading.Lock()
        self.current_frame_to_detect = None
        self.worker_thread = None
        self.last_snapshot_msg = ""
        self.last_snapshot_time = 0

    def clean_play_directory(self):
        """Safely clean up temporary audio files."""
        play_dir = 'play'
        if os.path.exists(play_dir):
            for f in os.listdir(play_dir):
                if f.endswith('.mp3'):
                    try:
                        os.remove(os.path.join(play_dir, f))
                    except Exception:
                        pass

    def _inference_worker(self):
        """Asynchronous background worker that runs YOLO without lagging the video stream."""
        while self.is_running:
            frame_copy = None
            with self.detection_lock:
                if self.current_frame_to_detect is not None:
                    frame_copy = self.current_frame_to_detect.copy()
                    self.current_frame_to_detect = None

            if frame_copy is None:
                time.sleep(0.01)
                continue

            h, w = frame_copy.shape[:2]
            try:
                # Prepare Darknet blob (416x416)
                blob = cv.dnn.blobFromImage(frame_copy, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)
                
                boxes, confs, cids = listBoundingBoxes(outs, h, w, self.conf_threshold)
                ids = ()
                if len(boxes) > 0:
                    ids = cv.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.nms_threshold)
                
                # Extract detected class names
                classes = []
                if ids is not None and len(ids) > 0:
                    flat_ids = np.array(ids).flatten()
                    classes = [self.class_labels[cids[i]] for i in flat_ids if i < len(cids)]

                # Update shared detection results
                with self.detection_lock:
                    self.latest_boxes = boxes
                    self.latest_confs = confs
                    self.latest_class_ids = cids
                    self.latest_ids = ids
                    self.latest_detected_classes = classes

                # Trigger voice feedback with smart debouncing
                if classes and self.enable_voice and not self.voice.muted:
                    self.voice.announce_objects(classes)

            except Exception as e:
                print(f"[ERROR in Inference Worker] {e}")

    def run_webcam(self, camera_index=0):
        """Run real-time detection from webcam with responsive display."""
        print(f"[INFO] Opening camera {camera_index}...")
        cap = cv.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"[WARN] Camera {camera_index} failed to open. Trying camera 1...")
            cap = cv.VideoCapture(1)
            if not cap.isOpened():
                print("[ERROR] Could not open any webcam. Please check your camera connection or permissions.")
                return

        # Optimize camera capture resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

        fps_timer = time.time()
        fps_counter = 0
        fps = 0.0
        window_name = "YOLOv3 Real-Time Object Detection with Voice"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 960, 720)

        print("[INFO] Camera stream started! Controls:")
        print("       [Q] Quit | [M/V] Mute Voice | [E] Switch TTS Engine | [S] Screenshot | [SPACE] Pause")

        try:
            while self.is_running:
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[WARN] Failed to grab frame from camera.")
                        time.sleep(0.05)
                        continue

                    # Queue latest frame for async inference if worker is ready
                    with self.detection_lock:
                        if self.current_frame_to_detect is None:
                            self.current_frame_to_detect = frame

                    # Draw latest available bounding boxes on this frame
                    with self.detection_lock:
                        boxes = list(self.latest_boxes)
                        confs = list(self.latest_confs)
                        cids = list(self.latest_class_ids)
                        ids = self.latest_ids
                        curr_classes = list(self.latest_detected_classes)

                    annotated_frame, _ = labelsBoundingBoxes(
                        frame.copy(), boxes, confs, cids, ids, self.label_colors, self.class_labels
                    )
                    
                    # Compute rolling FPS
                    fps_counter += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 0.5:
                        fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_timer = time.time()

                    # Compose voice status display
                    if self.voice.muted or not self.enable_voice:
                        voice_info = "Voice: MUTED"
                    else:
                        engine_name = "Offline SAPI" if self.voice.engine_type == 'sapi' else "Online gTTS"
                        voice_info = f"Voice: ON [{engine_name}]"

                    # Draw HUD
                    annotated_frame = drawHUD(
                        annotated_frame,
                        fps=fps,
                        object_count=len(curr_classes),
                        voice_info=voice_info,
                        conf_thresh=self.conf_threshold,
                        is_paused=self.is_paused
                    )
                    
                    # Display snapshot confirmation message if recently taken
                    if time.time() - self.last_snapshot_time < 2.0:
                        cv.putText(annotated_frame, self.last_snapshot_msg, (20, 70),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv.LINE_AA)

                    last_displayed_frame = annotated_frame

                cv.imshow(window_name, last_displayed_frame)

                # Process user keystrokes
                key = cv.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("[INFO] Quitting application...")
                    break
                elif key in (ord('m'), ord('M'), ord('v'), ord('V')):  # Mute toggle
                    muted = self.voice.toggle_mute()
                    status = "MUTED" if muted else "ACTIVE"
                    print(f"[VOICE] Voice is now {status}")
                elif key in (ord('e'), ord('E')):  # Toggle TTS engine
                    new_engine = self.voice.toggle_engine()
                    print(f"[VOICE] Switched voice engine to: {new_engine.upper()}")
                elif key in (ord('s'), ord('S')):  # Save snapshot
                    path = saveSnapshot(last_displayed_frame)
                    self.last_snapshot_msg = f"Saved: {os.path.basename(path)}"
                    self.last_snapshot_time = time.time()
                    print(f"[SNAPSHOT] Saved image to {path}")
                elif key == ord(' '):  # Pause / Resume
                    self.is_paused = not self.is_paused
                    status = "PAUSED" if self.is_paused else "RESUMED"
                    print(f"[STREAM] {status}")
                elif key in (ord('+'), ord('=')):  # Increase confidence
                    self.conf_threshold = min(0.95, round(self.conf_threshold + 0.05, 2))
                    print(f"[CONF] Threshold increased to: {self.conf_threshold:.2f}")
                elif key in (ord('-'), ord('_')):  # Decrease confidence
                    self.conf_threshold = max(0.10, round(self.conf_threshold - 0.05, 2))
                    print(f"[CONF] Threshold decreased to: {self.conf_threshold:.2f}")

        finally:
            self.is_running = False
            cap.release()
            cv.destroyAllWindows()
            self.voice.stop()
            self.clean_play_directory()
            print("[INFO] Cleanup complete. Session ended.")

    def run_image(self, image_path):
        """Run detection on a single image file."""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file does not exist: {image_path}")
            return

        print(f"[INFO] Processing image: {image_path}...")
        image = cv.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not decode image at {image_path}")
            return

        h, w = image.shape[:2]
        annotated_image, detected_classes, _, _, _, _ = detectObject(
            self.net, self.output_layers, h, w, image, self.label_colors, self.class_labels,
            conf_threshold=self.conf_threshold, nms_threshold=self.nms_threshold
        )

        print(f"[INFO] Objects detected ({len(detected_classes)}): {', '.join(detected_classes) if detected_classes else 'None'}")
        
        # Announce detected objects
        if detected_classes and self.enable_voice:
            text = self.voice.format_speech_text(detected_classes)
            print(f"[VOICE] Announcing: {text}")
            self.voice.say(text)

        # Draw HUD
        voice_info = "Voice: ON" if self.enable_voice else "Voice: OFF"
        annotated_image = drawHUD(
            annotated_image, fps=0.0, object_count=len(detected_classes),
            voice_info=voice_info, conf_thresh=self.conf_threshold
        )

        displayImage(annotated_image, f"Detection - {os.path.basename(image_path)}")
        self.voice.stop()

    def run_video(self, video_path):
        """Run detection on a video file."""
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file does not exist: {video_path}")
            return

        print(f"[INFO] Opening video: {video_path}...")
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

        window_name = f"Video Detection - {os.path.basename(video_path)}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        fps_timer = time.time()
        fps_counter = 0
        fps = 0.0

        try:
            while self.is_running and cap.isOpened():
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[INFO] Reached end of video.")
                        break

                    with self.detection_lock:
                        if self.current_frame_to_detect is None:
                            self.current_frame_to_detect = frame

                    with self.detection_lock:
                        boxes = list(self.latest_boxes)
                        confs = list(self.latest_confs)
                        cids = list(self.latest_class_ids)
                        ids = self.latest_ids
                        curr_classes = list(self.latest_detected_classes)

                    annotated_frame, _ = labelsBoundingBoxes(
                        frame.copy(), boxes, confs, cids, ids, self.label_colors, self.class_labels
                    )

                    fps_counter += 1
                    elapsed = time.time() - fps_timer
                    if elapsed >= 0.5:
                        fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_timer = time.time()

                    voice_info = "Voice: MUTED" if self.voice.muted else f"Voice: ON [{self.voice.engine_type.upper()}]"
                    annotated_frame = drawHUD(
                        annotated_frame, fps=fps, object_count=len(curr_classes),
                        voice_info=voice_info, conf_thresh=self.conf_threshold, is_paused=self.is_paused
                    )

                cv.imshow(window_name, annotated_frame)
                key = cv.waitKey(20) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('m'), ord('M'), ord('v'), ord('V')):
                    self.voice.toggle_mute()
                elif key == ord(' '):
                    self.is_paused = not self.is_paused
                elif key in (ord('s'), ord('S')):
                    saveSnapshot(annotated_frame)

        finally:
            self.is_running = False
            cap.release()
            cv.destroyAllWindows()
            self.voice.stop()
            self.clean_play_directory()


# Legacy interface functions for backward compatibility
def loadLibraries():
    pass

def deleteDirectory():
    app = YOLOVoiceApp()
    app.clean_play_directory()

def detectFromVideo():
    app = YOLOVoiceApp()
    app.run_webcam(0)

def detectFromImage(imagename):
    app = YOLOVoiceApp()
    app.run_image(imagename)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time YOLOv3 Object Detection with Voice Feedback")
    parser.add_argument('--gui', action='store_true', help="Launch the Desktop GUI application")
    parser.add_argument('--image', type=str, default=None, help="Path to input image file")
    parser.add_argument('--video', type=str, default=None, help="Path to input video file")
    parser.add_argument('--cam', type=int, default=0, help="Webcam camera index (default: 0)")
    parser.add_argument('--conf', type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument('--nms', type=float, default=0.3, help="NMS threshold (default: 0.3)")
    parser.add_argument('--engine', type=str, choices=['sapi', 'gtts'], default='sapi',
                        help="Text-to-speech engine: sapi (offline, Windows native) or gtts (online Google)")
    parser.add_argument('--no-voice', action='store_true', help="Disable voice announcements")
    parser.add_argument('--cooldown', type=float, default=4.0, help="Speech announcement cooldown in seconds")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    if args.gui:
        # Launch modern GUI
        import App
        App.main()
    else:
        app = YOLOVoiceApp(
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            voice_engine_type=args.engine,
            enable_voice=not args.no_voice,
            voice_cooldown=args.cooldown
        )
        
        if args.image:
            app.run_image(args.image)
        elif args.video:
            app.run_video(args.video)
        else:
            app.run_webcam(args.cam)
