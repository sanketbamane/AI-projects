# Voice Interview Bot (Python)

This project contains a simple voice-based interview bot that can:

- Ask role-specific interview questions (AI Engineer, ML Researcher, Data Scientist).
- Accept answers via microphone (SpeechRecognition/Vosk/PocketSphinx) or text fallback.
- Respond using offline TTS via pyttsx3.
- Score answers using a keyword-and-heuristic based scorer and save session logs to JSON.

Quick start (Windows cmd.exe):

1. Create and activate a virtual environment:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies. For PyAudio on Windows it's easiest to use pipwin:

```cmd
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```

3. (Optional) If you want offline ASR with Vosk, install the small model and put it in `model/`.
Download models from https://alphacephei.com/vosk/models and extract to `model` folder.

4. Run the interview (text-only mode if you don't have a microphone):

```cmd
python main.py --text-only
```

Or with microphone (default):

```cmd
python main.py --role "AI Engineer"
```

Notes and troubleshooting:

- If `pyttsx3` fails to initialize, the CLI will print text instead of speaking.
- If `SpeechRecognition` cannot access a microphone or recognition fails, the CLI will ask you to type answers.
- For better offline speech recognition, install `vosk` and a model; the AudioManager will attempt to use Vosk when available.

License: MIT-style permissive use.
