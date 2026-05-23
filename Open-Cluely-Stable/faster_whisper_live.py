#!/usr/bin/env python3
"""
Live Speech-to-Text using Faster-Whisper for Electron App
Streams results to stdout as JSON for real-time display
Uses sounddevice for capture and faster-whisper for high-accuracy transcription
"""

import os
import sys
import json
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Prevent standard error from being cluttered with logs
import logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

class WhisperLiveTranscriber:
    def __init__(self):
        # Configuration
        self.model_size = os.environ.get("WHISPER_MODEL", "base.en")
        self.device = os.environ.get("WHISPER_DEVICE", "cpu")
        self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        
        self.send_status("loading", f"Loading Faster-Whisper model ({self.model_size})...")
        
        try:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.expanduser("~"), ".faster_whisper_models")
            )
        except Exception as e:
            self.send_error(f"Failed to load model: {e}")
            sys.exit(1)

        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.should_exit = False
        
        # Buffering logic
        self.audio_buffer = np.array([], dtype=np.float32)
        self.max_buffer_len = self.sample_rate * 30  # 30 seconds max
        self.min_transcribe_len = int(self.sample_rate * 0.5) # Transcribe every 0.5s
        
        self._lock = threading.RLock()
        self.send_status("ready", "Faster-Whisper ready!")

    def send_status(self, status, message):
        output = {"type": "status", "status": status, "message": message}
        print(json.dumps(output), flush=True)

    def send_partial(self, text):
        output = {"type": "partial", "text": text}
        print(json.dumps(output), flush=True)

    def send_final(self, text):
        output = {"type": "final", "text": text}
        print(json.dumps(output), flush=True)

    def send_error(self, error):
        output = {"type": "error", "error": str(error)}
        print(json.dumps(output), flush=True)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.send_error(f"Audio error: {status}")
        with self._lock:
            if self.is_listening:
                self.audio_queue.put(indata.copy().flatten())

    def start_listening(self):
        with self._lock:
            if self.is_listening: return
            self.is_listening = True
            self.audio_buffer = np.array([], dtype=np.float32)
        self.send_status("listening", "Whisper Listening...")

    def stop_listening(self):
        with self._lock:
            if not self.is_listening: return
            self.is_listening = False
            # Finalize whatever is in the buffer
            if len(self.audio_buffer) > 0:
                self.transcribe_buffer(is_final=True)
            self.audio_buffer = np.array([], dtype=np.float32)
        self.send_status("stopped", "Stopped listening")

    def transcribe_buffer(self, is_final=False):
        if len(self.audio_buffer) < self.sample_rate * 0.2: # Need at least 200ms
            return

        try:
            # transcribe() returns a generator for segments
            segments, info = self.model.transcribe(
                self.audio_buffer,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                language="en" if ".en" in self.model_size else None
            )
            
            text = "".join([s.text for s in segments]).strip()
            
            if text:
                if is_final:
                    self.send_final(text)
                else:
                    self.send_partial(text)
            
            # Simple VAD-like logic: if it's a final result or we hit max buffer, clear it
            if is_final or len(self.audio_buffer) >= self.max_buffer_len:
                self.audio_buffer = np.array([], dtype=np.float32)
                
        except Exception as e:
            self.send_error(f"Transcription error: {e}")

    def _stdin_loop(self):
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd == "pause":
                    self.stop_listening()
                elif cmd == "resume":
                    self.start_listening()
                elif cmd == "quit":
                    self.should_exit = True
                    break
        except EOFError:
            self.should_exit = True

    def run(self):
        try:
            self.send_status("starting", "Opening audio stream...")
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype="float32"
            )
            
            with stream:
                self.send_status("stream_open", f"Stream opened successfully at {self.sample_rate}Hz")
                threading.Thread(target=self._stdin_loop, daemon=True).start()
                self.start_listening()
                
                last_transcribe_time = 0
                
                while not self.should_exit:
                    try:
                        # Get data from queue
                        data = self.audio_queue.get(timeout=0.1)
                        with self._lock:
                            self.audio_buffer = np.append(self.audio_buffer, data)
                            
                            # Periodic partial transcription
                            current_len = len(self.audio_buffer)
                            if current_len >= self.min_transcribe_len:
                                # We check for silence at the end to decide if we should finalize
                                # For now, let's just do partials and rely on the UI/Model to handle it
                                self.transcribe_buffer(is_final=False)
                                
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.send_error(f"Loop error: {e}")
                        
        except Exception as e:
            self.send_error(f"Fatal error: {e}")
            try:
                devices = sd.query_devices()
                self.send_status("info", f"Available devices: {len(devices)}")
            except:
                pass
            sys.exit(1)

def main():
    transcriber = WhisperLiveTranscriber()
    transcriber.run()

if __name__ == "__main__":
    main()
