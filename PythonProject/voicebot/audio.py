"""Audio input manager: microphone listening with fallbacks to text.

This module uses SpeechRecognition to capture from microphone. If unavailable,
it falls back to typed input. Vosk support is attempted if installed.
"""
from typing import Optional
import logging

log = logging.getLogger(__name__)


class AudioManager:
    def __init__(self, recognizer_name: Optional[str] = None):
        # recognizer_name can be 'vosk' to prefer Vosk if available
        self.recognizer_name = recognizer_name
        try:
            import speech_recognition as sr

            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.microphone = None
            self._has_microphone = True
            try:
                self.microphone = sr.Microphone()
            except Exception as e:
                # Microphone absence is not an error for text-only usage; debug-level log
                log.debug('Microphone not available: %s', e)
                self._has_microphone = False
        except Exception as e:
            # SpeechRecognition is optional; log at debug level to avoid noisy startup messages
            log.debug('SpeechRecognition not available: %s', e)
            self.sr = None
            self.recognizer = None
            self.microphone = None
            self._has_microphone = False

    def has_microphone(self) -> bool:
        return bool(self._has_microphone)

    def listen(self, prompt: str = None, timeout: Optional[int] = 8, phrase_time_limit: Optional[int] = 30) -> str:
        """Listen from the microphone and return recognized text.

        Falls back to text input if microphone or recognizer fails.
        """
        if prompt:
            print(prompt)
        # If SpeechRecognition or microphone is unavailable, ask for typed input with retries
        if not self.sr or not self._has_microphone:
            attempts = 3
            for i in range(attempts):
                try:
                    ans = input('(text input) Your answer: ').strip()
                except EOFError:
                    # EOF (e.g., Ctrl+Z) — treat as empty
                    ans = ''
                if ans:
                    return ans
                print(f'Empty answer received. Please type your answer ({i+1}/{attempts}).')
            # Return empty string after retries
            return ''

        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except self.sr.WaitTimeoutError:
                print('No speech detected (timeout). Please type your answer:')
                # Reprompt for typed input
                try:
                    return input('(text input) Your answer: ').strip()
                except EOFError:
                    return ''
            except Exception as e:
                # Non-fatal: fallback to typed input; log at debug level
                log.debug('Microphone listening failed: %s', e)
                try:
                    return input('(text input) Your answer: ').strip()
                except EOFError:
                    return ''

        # Try Vosk if requested and installed
        if self.recognizer_name == 'vosk':
            try:
                from vosk import Model, KaldiRecognizer
                import json

                # Attempt to use small model if present in current folder 'model'
                model = Model('model')
                rec = KaldiRecognizer(model, 16000)
                data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                if rec.AcceptWaveform(data):
                    res = rec.Result()
                    j = json.loads(res)
                    return j.get('text', '')
            except Exception:
                # fall through to other recognizers
                pass

        # Default: SpeechRecognition Google Web API (online) then Sphinx (offline) fallback
        try:
            # First try Google (may require internet)
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception:
            try:
                # Try PocketSphinx offline if installed
                text = self.recognizer.recognize_sphinx(audio)
                return text
            except Exception:
                print('Could not understand audio. Please type your answer:')
                return input('(text input) Your answer: ')
