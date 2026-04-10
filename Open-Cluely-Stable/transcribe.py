import whisper
import sys
import json

audio_file = sys.argv[1]

try:
    model = whisper.load_model("base")

    result = model.transcribe(audio_file)

    print(json.dumps({
        "success": True,
        "text": result["text"]
    }))

except Exception as e:
    print(json.dumps({
        "success": False,
        "error": str(e)
    }))