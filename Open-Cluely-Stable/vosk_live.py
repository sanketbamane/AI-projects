#!/usr/bin/env python3
"""
Live Speech-to-Text using Vosk for Electron App
Streams results to stdout as JSON for real-time display
Model loads once and stays in memory — use stdin commands: pause / resume / quit
"""

import sounddevice as sd
import queue
import json
import sys
import os
import zipfile
import requests
import threading
from pathlib import Path
from vosk import Model, KaldiRecognizer


class VoskLiveTranscriber:
    def __init__(self):
        """Initialize Vosk with the accurate US English model"""

        self.model_name = "vosk-model-en-us-0.22"
        self.model_url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
        self.model_dir = Path.home() / ".vosk_models"
        self.model_path = self.model_dir / self.model_name

        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            self.send_status("downloading", "Downloading Vosk model (one-time, ~1.8GB)...")
            self.download_model()

        self.send_status("loading", "Loading Vosk model...")
        self.model = Model(str(self.model_path))
        self.sample_rate = 16000
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.should_exit = False
        self.stream = None
        self._lock = threading.RLock()

        self._input_device = self._resolve_input_device()
        self.send_status("ready", "Vosk ready!")

    def _resolve_input_device(self):
        """None = PortAudio default. Or set VOSK_INPUT_DEVICE to index (e.g. 1) or name substring."""
        raw = os.environ.get("VOSK_INPUT_DEVICE", "").strip()
        try:
            default = sd.query_devices(kind="input")
            hint = default["name"] if default else "default"
            self.send_status("device", f"Default input device: {hint}")
        except Exception:
            pass

        if not raw:
            return None

        if raw.isdigit():
            idx = int(raw)
            try:
                info = sd.query_devices(idx)
                self.send_status("device", f"Using input device [{idx}]: {info['name']}")
                return idx
            except Exception as e:
                self.send_error(f"Invalid VOSK_INPUT_DEVICE index {idx}: {e}")
                return None

        try:
            devices = sd.query_devices()
            raw_lower = raw.lower()
            for i, d in enumerate(devices):
                if d["max_input_channels"] < 1:
                    continue
                if raw_lower in d["name"].lower():
                    self.send_status("device", f"Using input device [{i}]: {d['name']}")
                    return i
            self.send_error(f"No input device name contains '{raw}'. Check VOSK_INPUT_DEVICE.")
        except Exception as e:
            self.send_error(f"Device lookup failed: {e}")
        return None

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

    def download_model(self):
        try:
            zip_path = self.model_dir / f"{self.model_name}.zip"
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            self.send_status("downloading", f"Downloading: {percent:.1f}%")

            self.send_status("extracting", "Extracting model...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.model_dir)

            zip_path.unlink()
            self.send_status("ready", "Model ready!")

        except Exception as e:
            self.send_error(f"Download failed: {e}")
            sys.exit(1)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            self.send_error(f"Audio error: {status}")
        with self._lock:
            listening = self.is_listening
        if listening:
            self.audio_queue.put(bytes(indata))

    def _drain_audio_queue(self):
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

    def start_listening(self):
        with self._lock:
            if self.is_listening:
                return

            self.is_listening = True
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)

        self._drain_audio_queue()
        self.send_status("listening", "Listening...")

    def stop_listening(self):
        with self._lock:
            if not self.is_listening:
                return

            self.is_listening = False

            try:
                final_result = json.loads(self.recognizer.FinalResult())
                text = final_result.get("text", "").strip()
                if text:
                    self.send_final(text)
            except Exception:
                pass

        self._drain_audio_queue()
        self.send_status("stopped", "Stopped listening")

    def _stdin_loop(self):
        try:
            for line in sys.stdin:
                if self.should_exit:
                    break
                cmd = line.strip().lower()
                if cmd == "pause":
                    self.stop_listening()
                elif cmd == "resume":
                    self.start_listening()
                elif cmd == "quit":
                    self.should_exit = True
                    break
        except Exception:
            pass

    def run(self):
        try:
            self.send_status("starting", "Opening audio stream...")
            stream_kw = dict(
                samplerate=self.sample_rate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=self.audio_callback,
                latency="low",
            )
            if self._input_device is not None:
                stream_kw["device"] = self._input_device

            self.stream = sd.RawInputStream(**stream_kw)

            with self.stream:
                self.send_status("stream_open", f"Stream opened successfully at {self.sample_rate}Hz")
                threading.Thread(target=self._stdin_loop, daemon=True).start()
                self.start_listening()

                while not self.should_exit:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    with self._lock:
                        if not self.is_listening:
                            continue
                        try:
                            if self.recognizer.AcceptWaveform(data):
                                result = json.loads(self.recognizer.Result())
                                text = result.get("text", "").strip()
                                if text:
                                    self.send_final(text)
                            else:
                                result = json.loads(self.recognizer.PartialResult())
                                text = result.get("partial", "").strip()
                                if text:
                                    self.send_partial(text)
                        except Exception as e:
                            self.send_error(f"Recognize error: {e}")

        except KeyboardInterrupt:
            self.stop_listening()
        except Exception as e:
            self.send_error(f"Fatal error: {e}")
            try:
                devices = sd.query_devices()
                self.send_status("info", f"Available devices: {len(devices)}")
            except:
                pass
            sys.exit(1)


def main():
    try:
        transcriber = VoskLiveTranscriber()
        transcriber.run()
    except Exception as e:
        print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
