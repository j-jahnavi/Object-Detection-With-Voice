# VisionVoice: Real-Time YOLOv3 Object Detection with Voice Feedback

VisionVoice is an advanced, real-time object detection and recognition system using the YOLOv3 deep learning model with synchronized text-to-speech (TTS) voice announcements.

---

## 🚀 Key Improvements & Developments

### 1. Fixed the `urllib3` / OpenSSL Python 3.7 Crash
- **Problem**: Python 3.7's SSL module on Windows was built with OpenSSL 1.1.0h, which causes `urllib3 v2.0+` to raise `ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+`.
- **Solution**: Pinned and downgraded to `urllib3<2.0.0` (`1.26.20`), eliminating the crash completely.

### 2. Multi-Engine Voice System (`voice_engine.py`)
- **Offline SAPI5 (Windows Native)**: Instant, zero-latency speech with no internet requirement.
- **Online Google TTS (`gTTS`)**: Optional online voice mode with safe file caching and auto-cleanup.
- **Thread-safe Worker Queue**: Decoupled from the video loop via a background queue (`queue.Queue`), ensuring the camera stream never stutters or drops frames.
- **Smart Debouncing**: Eliminates audio spamming. Only announces when:
  - A new object enters the scene.
  - A configurable cooldown timer (default: 4 seconds) has passed.
- **Natural Language Aggregation**: Converts raw detection lists (`['person', 'person', 'cup']`) into clean sentences: *"Detected 2 persons and a cup."*

### 3. Asynchronous Inference Pipeline (`Main.py`)
- Standard YOLOv3 inference takes ~2.7s on CPU.
- Running inference synchronously freezes the video stream to 0.35 FPS.
- VisionVoice uses an **asynchronous threaded pipeline**: the webcam feed renders smoothly at **30 FPS**, while detection boxes and speech updates run seamlessly in the background.

### 4. Modern Heads-Up Display (HUD) & Overlays (`ObjectDetection.py`)
- Contrast-enhanced badges displaying object name and confidence percentage.
- Real-time HUD banner displaying:
  - Stream status (`● LIVE STREAM` / `PAUSED`)
  - Live FPS counter
  - Detected objects count
  - Confidence threshold
  - Active audio engine & mute status
- One-click snapshot capture saved directly to the `snapshots/` folder.

### 5. Desktop GUI Application (`App.py`)
- Full graphical interface built with Tkinter:
  - Real-time video preview.
  - Source selection: Webcam, Image file, or Video file.
  - Interactive sliders for Confidence threshold, NMS threshold, and Speech cooldown.
  - Voice engine selector (SAPI5 vs gTTS) and mute toggle.
  - Live detection history table with timestamps and confidence scores.

---

## 🎮 Controls & Hotkeys (Webcam / Video Window)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit the application |
| `M` or `V` | Toggle Voice Mute / Unmute |
| `E` | Switch TTS Engine (`SAPI5` offline $\leftrightarrow$ `gTTS` online) |
| `S` | Save timestamped snapshot with detections |
| `SPACE` | Pause / Resume video stream |
| `+` / `=` | Increase confidence threshold (+0.05) |
| `-` / `_` | Decrease confidence threshold (-0.05) |

---

## 📁 Project Structure

```
ObjectDetection/
├── Main.py              # CLI & Core asynchronous webcam/video/image pipeline
├── App.py               # Modern Desktop GUI application
├── ObjectDetection.py   # YOLOv3 model inference, NMS, HUD, and bounding box drawing
├── voice_engine.py      # Thread-safe dual-engine TTS system with debouncing
├── run.bat              # Interactive launcher menu
├── requirements.txt     # Clean pip dependencies
├── model/
│   ├── yolov3.cfg       # YOLOv3 network configuration
│   ├── yolov3.weights   # Trained Darknet weights (248 MB)
│   └── yolov3-labels    # 80 COCO object classes
├── play/                # Temporary audio cache directory (auto-managed)
└── snapshots/           # Saved annotated frames
```

---

## ⚡ Quick Start

### 1. Launch Interactive Menu
Double-click `run.bat` or run:
```cmd
run.bat
```

### 2. Run Webcam Detection directly:
```cmd
python Main.py
```

### 3. Launch Desktop GUI:
```cmd
python App.py
```
*(or `python Main.py --gui`)*

### 4. Run on an Image or Video:
```cmd
python Main.py --image path/to/image.jpg
python Main.py --video path/to/video.mp4
```

### 5. Custom Options:
```cmd
python Main.py --cam 0 --conf 0.6 --engine sapi --cooldown 5.0
```
