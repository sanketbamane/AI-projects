import sys
import os

print("Checking core dependencies...")
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
except Exception as e:
    print(f"Numpy import failed: {e}")

try:
    import sounddevice
    print(f"Sounddevice version: {sounddevice.__version__}")
except Exception as e:
    print(f"Sounddevice import failed: {e}")

print("\nChecking STT Engines...")
try:
    import faster_whisper
    print("Faster-Whisper: OK")
except Exception as e:
    print(f"Faster-Whisper: FAILED ({e})")

try:
    import vosk
    print("Vosk: OK")
except Exception as e:
    print(f"Vosk: FAILED ({e})")

print("\nDIAGNOSTIC COMPLETE.")
