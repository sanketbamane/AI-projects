from faster_whisper import WhisperModel
import sys
import json
import os

audio_file = sys.argv[1]

try:
    model_size = os.environ.get("WHISPER_MODEL", "base.en")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(audio_file, beam_size=5)
    text = "".join([segment.text for segment in segments])

    print(json.dumps({
        "success": True,
        "text": text.strip()
    }))

except Exception as e:
    print(json.dumps({
        "success": False,
        "error": str(e)
    }))