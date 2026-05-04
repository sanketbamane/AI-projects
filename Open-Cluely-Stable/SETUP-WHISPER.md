# Setup Faster-Whisper (High-Accuracy Local STT)

Faster-Whisper provides significantly better transcription accuracy than Vosk while remaining 100% offline.

## 1. Prerequisites

Ensure you have **Python 3.8+** installed.

## 2. Install Dependencies

Run the following command to install the required Python packages:

```bash
pip install faster-whisper sounddevice numpy
```

## 3. Configuration

Open your `.env` file and ensure the following variables are set:

```bash
STT_ENGINE=whisper
WHISPER_MODEL=base.en
```

### Model Options:
- `tiny.en` / `tiny`: Fastest, lowest memory usage.
- `base.en` / `base`: Good balance of speed and accuracy (Default).
- `small.en` / `small`: More accurate, requires more RAM/CPU.
- `medium.en` / `medium`: High accuracy, slower.
- `large-v3`: Highest accuracy, requires a GPU for real-time use.

## 4. Hardware Acceleration (Optional but Recommended)

If you have an **NVIDIA GPU**, you can make transcription nearly instant by using CUDA:

1. Install CUDA Toolkit and cuDNN.
2. Update your `.env`:
   ```bash
   WHISPER_DEVICE=cuda
   WHISPER_COMPUTE_TYPE=float16
   ```

## 5. First Run

The first time you start voice recognition, Faster-Whisper will automatically download the chosen model (approx. 140MB for `base.en`). This happens only once.

## 6. Troubleshooting

- **No audio captured**: Ensure your microphone is the default input device in Windows settings.
- **Slow transcription**: If your CPU is old, try `WHISPER_MODEL=tiny.en`.
- **ModuleNotFoundError**: Ensure you installed the packages in the same Python environment that Electron is calling.
