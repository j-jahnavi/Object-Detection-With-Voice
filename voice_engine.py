"""
Voice Engine Module for Real-Time Object Detection
Supports:
1. Windows Native SAPI5 (Offline, zero latency, ultra-reliable)
2. Google TTS (Online gTTS with safe file caching and cleanup)
Features:
- Thread-safe background worker queue (no video loop blocking)
- Smart debouncing & change detection (prevents audio spamming)
- Natural language phrasing (e.g., '2 persons and 1 bottle detected')
- Mute/Unmute toggle and engine switching
"""

import os
import sys
import time
import queue
import threading
from collections import Counter

# Try importing native Windows SAPI5 via pywin32
HAVE_SAPI = False
try:
    import win32com.client
    import pythoncom
    HAVE_SAPI = True
except Exception:
    HAVE_SAPI = False

# Try importing gTTS and playsound
HAVE_GTTS = False
try:
    from gtts import gTTS
    from playsound import playsound
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False


class VoiceEngine:
    def __init__(self, engine_type='sapi', cooldown_seconds=4.0, sound_dir='play'):
        self.engine_type = engine_type if (engine_type == 'sapi' and HAVE_SAPI) else ('gtts' if HAVE_GTTS else 'none')
        self.cooldown_seconds = cooldown_seconds
        self.sound_dir = sound_dir
        self.muted = False
        self.is_running = True
        
        # Audio file directory
        os.makedirs(self.sound_dir, exist_ok=True)
        
        # Debouncing state
        self.last_spoken_time = 0.0
        self.last_announced_classes = set()
        
        # Background speech queue (maxsize prevents accumulation of stale speech)
        self.speech_queue = queue.Queue(maxsize=3)
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def format_speech_text(self, class_list):
        """Convert a list of class names into natural English sentence."""
        if not class_list:
            return ""
        
        counts = Counter(class_list)
        parts = []
        for name, count in counts.items():
            if count == 1:
                # Add indefinite article 'a' or 'an'
                article = "an" if name[0].lower() in "aeiou" else "a"
                parts.append(f"{article} {name}")
            else:
                # Basic pluralization
                plural = name + "s" if not name.endswith("s") else name
                parts.append(f"{count} {plural}")
        
        if len(parts) == 1:
            return f"Detected {parts[0]}."
        elif len(parts) == 2:
            return f"Detected {parts[0]} and {parts[1]}."
        else:
            return f"Detected {', '.join(parts[:-1])}, and {parts[-1]}."

    def announce_objects(self, class_list, force=False):
        """
        Check if speech should be triggered.
        Announces if:
        1. Force is True, OR
        2. New unique object appeared, OR
        3. Cooldown has elapsed since last announcement
        """
        if self.muted or not class_list:
            return
        
        current_set = set(class_list)
        now = time.time()
        
        new_objects = current_set - self.last_announced_classes
        time_elapsed = now - self.last_spoken_time
        
        should_speak = force or (len(new_objects) > 0 and time_elapsed > 1.5) or (time_elapsed > self.cooldown_seconds)
        
        if should_speak:
            text = self.format_speech_text(class_list)
            self.last_spoken_time = now
            self.last_announced_classes = current_set
            self.say(text)

    def say(self, text):
        """Push text into speech queue, dropping old pending messages if queue is full."""
        if self.muted or not text:
            return
        
        try:
            # If queue is full, discard oldest item to avoid speech lag
            if self.speech_queue.full():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    pass
            self.speech_queue.put_nowait(text)
        except Exception:
            pass

    def toggle_mute(self):
        self.muted = not self.muted
        # Clear speech queue if muted
        if self.muted:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        return self.muted

    def toggle_engine(self):
        """Switch between offline (SAPI) and online (gTTS) engines."""
        if self.engine_type == 'sapi' and HAVE_GTTS:
            self.engine_type = 'gtts'
        elif HAVE_SAPI:
            self.engine_type = 'sapi'
        return self.engine_type

    def _speech_worker(self):
        """Dedicated background worker to sequentially process speech requests."""
        # Initialize COM in this thread for SAPI
        sapi_voice = None
        if HAVE_SAPI:
            try:
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 1  # Standard speech rate
            except Exception as e:
                sapi_voice = None

        file_idx = 0
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not text or self.muted:
                self.speech_queue.task_done()
                continue

            try:
                if self.engine_type == 'sapi' and sapi_voice is not None:
                    # Windows Native SAPI5 (Synchronous in worker thread, won't block UI/Video)
                    sapi_voice.Speak(text)
                elif HAVE_GTTS:
                    # Google TTS with rotating cache file
                    file_idx = (file_idx + 1) % 5
                    audio_path = os.path.join(self.sound_dir, f"voice_{file_idx}.mp3")
                    
                    # Remove previous file if exists
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except Exception:
                        pass
                    
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(audio_path)
                    playsound(audio_path)
                    
                    # Clean up
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except Exception:
                        pass
                elif sapi_voice is not None:
                    sapi_voice.Speak(text)
            except Exception as e:
                # Fallback to SAPI if gTTS fails (e.g. offline/network failure)
                if sapi_voice is not None and self.engine_type != 'sapi':
                    try:
                        sapi_voice.Speak(text)
                    except Exception:
                        pass
            finally:
                self.speech_queue.task_done()

        if HAVE_SAPI:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
